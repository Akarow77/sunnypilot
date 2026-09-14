#!/usr/bin/env python3
"""Compile the driving policy without camera warps for the remote worker."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np

from tinygrad.device import Device
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.onnx import OnnxRunner
from tinygrad.tensor import Tensor

from openpilot.selfdrive.modeld.compile_modeld import compile_jit, get_policy_npy_shapes, make_run_policy, read_file_chunked_to_disk
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.get_model_metadata import make_metadata_dict
from openpilot.selfdrive.modeld.helpers import dump_oob

from transport import WARPED_SHAPE


REMOTE_INPUTS = ['warped', 'img_q', 'big_img_q', 'feat_q', 'desire_q', 'packed_npy_inputs']


def make_remote_input_queues(metadata: dict, device: str):
  input_shapes = metadata['input_shapes']
  frame_skip = ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ
  img = input_shapes['img']
  features = input_shapes['features_buffer']
  desire = input_shapes['desire_pulse']
  frame_count = img[1] // 6
  image_queue_shape = (frame_skip * (frame_count - 1) + 1, 6, img[2], img[3])
  feature_dim = math.prod(features[2:])

  npy_shapes, npy_sizes = get_policy_npy_shapes(input_shapes)
  packed_policy = np.zeros(sum(npy_sizes), dtype=np.float32)
  npy = {name: value.reshape(shape) for (name, shape), value in zip(
    npy_shapes.items(), np.split(packed_policy, np.cumsum(npy_sizes[:-1])), strict=True)}
  warped = np.zeros(WARPED_SHAPE, dtype=np.uint8)

  queues = {
    'warped': Tensor(warped, device='NPY').realize(),
    'img_q': Tensor(np.zeros(image_queue_shape, dtype=np.uint8), device=device).contiguous().realize(),
    'big_img_q': Tensor(np.zeros(image_queue_shape, dtype=np.uint8), device=device).contiguous().realize(),
    'feat_q': Tensor(np.zeros((frame_skip * features[1], features[0], feature_dim), dtype=np.float32),
                     device=device).contiguous().realize(),
    'desire_q': Tensor(np.zeros((frame_skip * desire[1], desire[0], desire[2]), dtype=np.float32),
                       device=device).contiguous().realize(),
    'packed_npy_inputs': Tensor(packed_policy, device='NPY').realize(),
  }
  return queues, npy, {'warped': warped}


def main() -> None:
  parser = argparse.ArgumentParser(description="Compile a Metal policy-only JIT for remote inference")
  parser.add_argument('--onnx', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--benchmark-runs', type=int, default=10)
  args = parser.parse_args()
  if args.benchmark_runs < 1:
    parser.error('--benchmark-runs must be at least 1')

  model_path = read_file_chunked_to_disk(str(args.onnx))
  metadata = make_metadata_dict(model_path)
  policy = make_run_policy(OnnxRunner(model_path), metadata,
                           ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ)

  def run_remote(warped, img_q, big_img_q, feat_q, desire_q, packed_npy_inputs):
    return policy(warped.to(Device.DEFAULT), img_q, big_img_q, feat_q, desire_q, packed_npy_inputs)

  def make_queues(device: str):
    return make_remote_input_queues(metadata, device)

  jit = TinyJit(run_remote, prune=True)
  jit = compile_jit(jit, REMOTE_INPUTS, make_queues, args.benchmark_runs)
  artifact = {
    'metadata': metadata,
    'input_devices': {'model': Device.DEFAULT},
    'run_policy': jit,
    'accelerator': {
      'model_sha256': hashlib.sha256(Path(args.onnx).read_bytes()).hexdigest(),
    },
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open('wb') as output_file:
    dump_oob(artifact, output_file)
  print(f"Policy artifact: {args.output} ({args.output.stat().st_size / 1e6:.2f} MB)")


if __name__ == '__main__':
  main()
