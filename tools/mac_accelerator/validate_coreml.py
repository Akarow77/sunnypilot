#!/usr/bin/env python3
"""Compare recurrent Core ML outputs with the traced FP16 source model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import coremltools as ct
import numpy as np
import torch

from coreml_inference_server import CoreMLPolicySession
from benchmark_coreml import COMPUTE_UNITS
from transport import POLICY_INPUTS, WARPED_SHAPE


def error_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
  error = np.abs(candidate - reference)
  reference_rms = float(np.sqrt(np.mean(np.square(reference, dtype=np.float64))))
  error_rms = float(np.sqrt(np.mean(np.square(error, dtype=np.float64))))
  return {
    'max_abs': float(np.max(error)),
    'mean_abs': float(np.mean(error)),
    'nrmse': error_rms / max(reference_rms, 1e-12),
    'p99_abs': float(np.percentile(error, 99)),
  }


class TorchAdapter:
  def __init__(self, model):
    self.model = model

  def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    with torch.inference_mode():
      output = self.model(
        torch.from_numpy(inputs['img'].astype(np.uint8)),
        torch.from_numpy(inputs['big_img'].astype(np.uint8)),
        torch.from_numpy(inputs['desire_pulse'].astype(np.float16)),
        torch.from_numpy(inputs['traffic_convention'].astype(np.float16)),
        torch.from_numpy(inputs['action_t'].astype(np.float16)),
        torch.from_numpy(inputs['features_buffer'].astype(np.float16)),
      ).float().numpy()
    return {'output': output}


def main() -> None:
  parser = argparse.ArgumentParser(description='Validate Core ML against the source FP16 trace')
  parser.add_argument('model', type=Path)
  parser.add_argument('--torch-reference', type=Path, required=True)
  parser.add_argument('--metadata', type=Path, required=True)
  parser.add_argument('--samples', type=int, default=3)
  parser.add_argument('--compute-unit', choices=COMPUTE_UNITS, default='cpu-and-ne',
                      help='default matches the deployed Core ML worker')
  parser.add_argument('--warps', type=Path, help='optional .npy array shaped [frames,2,6,128,256]')
  parser.add_argument('--json-output', type=Path)
  args = parser.parse_args()
  if args.samples < 1:
    parser.error('--samples must be positive')

  with args.metadata.open('rb') as metadata_file:
    metadata = pickle.load(metadata_file)
  expected_shape = tuple(metadata['output_shapes']['outputs'])

  coreml_model = ct.models.MLModel(str(args.model), compute_units=COMPUTE_UNITS[args.compute_unit])
  coreml_output_name = coreml_model.get_spec().description.output[0].name
  torch_model = torch.jit.load(args.torch_reference).eval().half()
  rng = np.random.default_rng(20260914)
  coreml_session = CoreMLPolicySession(coreml_model, metadata, coreml_output_name, frame_skip=2)
  torch_session = CoreMLPolicySession(TorchAdapter(torch_model), metadata, 'output', frame_skip=2,
                                      output_dtype='<f4')
  warps = np.load(args.warps) if args.warps is not None else None
  if warps is not None and (warps.shape[1:] != WARPED_SHAPE or len(warps) < args.samples):
    raise RuntimeError(f'invalid warp array: {warps.shape}')
  sample_reports = []
  policy = POLICY_INPUTS.pack(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.1, 0.1)

  for frame in range(args.samples):
    warped = warps[frame] if warps is not None else rng.integers(0, 256, WARPED_SHAPE, dtype=np.uint8)
    payload = warped.tobytes() + policy
    coreml_output = np.frombuffer(coreml_session.infer(payload), dtype='<f2').astype(np.float32).reshape(expected_shape)
    torch_output = np.frombuffer(torch_session.infer(payload), dtype='<f4').reshape(expected_shape)
    if coreml_output.shape != expected_shape or torch_output.shape != expected_shape:
      raise RuntimeError(f'output shape mismatch: Core ML {coreml_output.shape}, Torch {torch_output.shape}')
    if not np.all(np.isfinite(coreml_output)) or not np.all(np.isfinite(torch_output)):
      raise RuntimeError('non-finite validation output')

    sample_report = {
      'frame': frame,
      **error_metrics(coreml_output, torch_output),
      'fields': {
        name: error_metrics(coreml_output[:, output_slice], torch_output[:, output_slice])
        for name, output_slice in metadata['output_slices'].items()
      },
    }
    sample_reports.append(sample_report)

  report = {
    'validationStatus': 'measurement-only; no numerical acceptance threshold established',
    'computeUnit': args.compute_unit,
    'inputSource': str(args.warps) if args.warps is not None else 'synthetic',
    'reference': str(args.torch_reference),
    'max_abs': max(sample['max_abs'] for sample in sample_reports),
    'max_nrmse': max(sample['nrmse'] for sample in sample_reports),
    'field_maxima': {
      name: {
        metric: max(sample['fields'][name][metric] for sample in sample_reports)
        for metric in ('max_abs', 'mean_abs', 'nrmse', 'p99_abs')
      }
      for name in metadata['output_slices']
    },
    'samples': sample_reports,
  }
  print(json.dumps(report, indent=2, sort_keys=True))
  if args.json_output is not None:
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
  main()
