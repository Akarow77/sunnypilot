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


def summarize_log(path: Path, deadline_ms: float = 55.0) -> dict:
  lines = path.read_text().splitlines()
  all_rows = []
  truncated_tail = False
  for index, line in enumerate(lines):
    if not line.strip():
      continue
    try:
      all_rows.append(json.loads(line))
    except json.JSONDecodeError:
      if index != len(lines) - 1:
        raise
      truncated_tail = True
  rows = [row for row in all_rows if 'copyToOutputMs' in row or 'warpToOutputMs' in row]
  failures = [row['error'] for row in all_rows if 'error' in row]
  warp_to_output = [float(row.get('copyToOutputMs', row.get('warpToOutputMs'))) for row in rows]
  round_trip = [float(row['roundTripMs']) for row in rows]
  inference = [float(row['inferenceMs']) for row in rows]
  differences = [abs(float(row['curvatureDifference'])) for row in rows
                 if row.get('comparable', False) and row.get('curvatureDifference') is not None]
  report = {
    'path': str(path),
    'frames': len(rows),
    'failure': failures[-1] if failures else None,
    'failureCount': len(failures),
    'truncatedTail': truncated_tail,
    'deadlineMs': deadline_ms,
    'completedFrameDeadlineMisses': sum(row.get('deadlineMiss', value > deadline_ms) for row, value in zip(rows, warp_to_output, strict=True)),
    'timeoutOrDeadlineFailures': sum('deadline' in failure or 'timed out' in failure for failure in failures),
    'qualifiedFrames': sum(row.get('qualified', False) for row in rows),
    'contextResets': sum(row.get('event') == 'contextReset' for row in all_rows),
    'missingCameraFrames': sum(row.get('missingFrames', 0) for row in rows),
    'producerDrops': max((int(row.get('producerDrops', 0)) for row in rows), default=0),
    'copyToOutputMs': summary(warp_to_output),
    'captureToOutputMs': summary([row['captureToOutputMs'] for row in rows if 'captureToOutputMs' in row]),
    'roundTripMs': summary(round_trip),
    'inferenceMs': summary(inference),
    'absoluteCurvatureDifference': summary(differences),
  }
  return report


def main() -> None:
  parser = argparse.ArgumentParser(description='summarize a Mac accelerator live-shadow log')
  parser.add_argument('log', type=Path)
  parser.add_argument('--deadline-ms', type=float, default=55.0)
  args = parser.parse_args()
  report = summarize_log(args.log, args.deadline_ms)
  print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
  main()
