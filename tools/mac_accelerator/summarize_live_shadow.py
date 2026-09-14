#!/usr/bin/env python3
"""Summarize a live-shadow JSONL log copied from a comma 3X."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


def percentile(values: list[float], q: float) -> float:
  ordered = sorted(values)
  position = (len(ordered) - 1) * q / 100.0
  low = math.floor(position)
  high = math.ceil(position)
  if low == high:
    return ordered[low]
  return ordered[low] * (high - position) + ordered[high] * (position - low)


def summary(values: list[float]) -> dict[str, float | int]:
  if not values:
    return {'count': 0}
  return {
    'count': len(values),
    'mean': statistics.fmean(values),
    'p50': percentile(values, 50),
    'p95': percentile(values, 95),
    'p99': percentile(values, 99),
    'max': max(values),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description='summarize a Mac accelerator live-shadow log')
  parser.add_argument('log', type=Path)
  parser.add_argument('--deadline-ms', type=float, default=55.0)
  args = parser.parse_args()

  all_rows = [json.loads(line) for line in args.log.read_text().splitlines() if line.strip()]
  rows = [row for row in all_rows if 'warpToOutputMs' in row]
  failures = [row['error'] for row in all_rows if 'error' in row]
  warp_to_output = [float(row['warpToOutputMs']) for row in rows]
  round_trip = [float(row['roundTripMs']) for row in rows]
  inference = [float(row['inferenceMs']) for row in rows]
  differences = [abs(float(row['curvatureDifference'])) for row in rows if row.get('curvatureDifference') is not None]
  report = {
    'path': str(args.log),
    'frames': len(rows),
    'failure': failures[-1] if failures else None,
    'deadlineMs': args.deadline_ms,
    'deadlineMisses': sum(value > args.deadline_ms for value in warp_to_output),
    'producerDrops': max((int(row.get('producerDrops', 0)) for row in rows), default=0),
    'warpToOutputMs': summary(warp_to_output),
    'roundTripMs': summary(round_trip),
    'inferenceMs': summary(inference),
    'absoluteCurvatureDifference': summary(differences),
  }
  print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
  main()
