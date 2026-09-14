#!/usr/bin/env python3
"""Benchmark a converted driving model with completed Core ML predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import time

import coremltools as ct
import numpy as np


COMPUTE_UNITS = {
  'all': ct.ComputeUnit.ALL,
  'cpu-and-gpu': ct.ComputeUnit.CPU_AND_GPU,
  'cpu-and-ne': ct.ComputeUnit.CPU_AND_NE,
  'cpu-only': ct.ComputeUnit.CPU_ONLY,
}


def make_inputs(rng: np.random.Generator) -> dict[str, np.ndarray]:
  return {
    'img': rng.integers(0, 256, (1, 12, 128, 256), dtype=np.uint8).astype(np.float32),
    'big_img': rng.integers(0, 256, (1, 12, 128, 256), dtype=np.uint8).astype(np.float32),
    'desire_pulse': np.zeros((1, 33, 8), dtype=np.float32),
    'traffic_convention': np.array([[1.0, 0.0]], dtype=np.float32),
    'action_t': np.array([[0.1, 0.1]], dtype=np.float32),
    'features_buffer': np.zeros((1, 32, 32, 512), dtype=np.float32),
  }


def percentile(values: list[float], quantile: float) -> float:
  return float(np.percentile(np.asarray(values), quantile))


def main() -> None:
  parser = argparse.ArgumentParser(description='Benchmark completed Core ML driving-model inference')
  parser.add_argument('model', type=Path)
  parser.add_argument('--metadata', type=Path, required=True)
  parser.add_argument('--compute-unit', choices=COMPUTE_UNITS, default='all')
  parser.add_argument('--warmup', type=int, default=5)
  parser.add_argument('--runs', type=int, default=100)
  parser.add_argument('--frequency', type=float, default=0.0,
                      help='pace predictions at this frequency; zero runs as fast as possible')
  parser.add_argument('--duration', type=float,
                      help='derive run count from frequency and this duration in seconds')
  parser.add_argument('--deadline-ms', type=float, default=50.0)
  parser.add_argument('--json-output', type=Path)
  parser.add_argument('--require-zero-misses', action='store_true')
  args = parser.parse_args()
  if args.duration is not None:
    if args.duration <= 0 or args.frequency <= 0:
      parser.error('--duration requires a positive --frequency')
    args.runs = round(args.duration * args.frequency)
  if args.warmup < 1 or args.runs < 1 or args.frequency < 0 or args.deadline_ms <= 0:
    parser.error('warmup, runs, frequency, and deadline must be non-negative or positive as appropriate')

  with args.metadata.open('rb') as metadata_file:
    metadata = pickle.load(metadata_file)
  hidden_slice = metadata['output_slices']['hidden_state']
  expected_shape = tuple(metadata['output_shapes']['outputs'])

  load_started = time.perf_counter()
  model = ct.models.MLModel(str(args.model), compute_units=COMPUTE_UNITS[args.compute_unit])
  load_seconds = time.perf_counter() - load_started
  output_name = model.get_spec().description.output[0].name
  inputs = make_inputs(np.random.default_rng(0))

  def predict() -> np.ndarray:
    result = np.asarray(model.predict(inputs)[output_name])
    if result.shape != expected_shape or not np.all(np.isfinite(result)):
      raise RuntimeError(f'invalid Core ML output: shape={result.shape}')
    feature_history = inputs['features_buffer']
    feature_history[:, :-1] = feature_history[:, 1:]
    feature_history[:, -1] = result[0, hidden_slice].reshape(feature_history[:, -1].shape)
    return result

  for _ in range(args.warmup):
    predict()
  durations = []
  schedule_lateness = []
  period = 1.0 / args.frequency if args.frequency else 0.0
  next_start = time.perf_counter()
  for _ in range(args.runs):
    if period:
      now = time.perf_counter()
      if now < next_start:
        time.sleep(next_start - now)
      schedule_lateness.append(max(0.0, time.perf_counter() - next_start) * 1000.0)
    started = time.perf_counter()
    predict()
    durations.append((time.perf_counter() - started) * 1000.0)
    next_start += period

  midpoint = max(1, len(durations) // 2)

  report = {
    'compute_unit': args.compute_unit,
    'deadline_ms': args.deadline_ms,
    'load_seconds': load_seconds,
    'max_ms': max(durations),
    'mean_ms': float(np.mean(durations)),
    'misses': sum(duration > args.deadline_ms for duration in durations),
    'first_half_mean_ms': float(np.mean(durations[:midpoint])),
    'second_half_mean_ms': float(np.mean(durations[midpoint:] or durations)),
    'frequency_hz': args.frequency,
    'p50_ms': percentile(durations, 50),
    'p95_ms': percentile(durations, 95),
    'p99_ms': percentile(durations, 99),
    'runs': args.runs,
  }
  if schedule_lateness:
    report['schedule_lateness_max_ms'] = max(schedule_lateness)
    report['schedule_lateness_p99_ms'] = percentile(schedule_lateness, 99)
  print(json.dumps(report, indent=2, sort_keys=True))
  if args.json_output is not None:
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
  if args.require_zero_misses and report['misses']:
    raise SystemExit(1)


if __name__ == '__main__':
  main()
