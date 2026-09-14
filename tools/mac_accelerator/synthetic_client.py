#!/usr/bin/env python3
"""Fail-closed 20 Hz synthetic client suitable for running on a comma 3X."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import random
import statistics
import time

from accelerator_client import AcceleratorClient
from accelerator_protocol import read_auth_key
from transport import POLICY_INPUTS, WARPED_BYTES
from ui_state import make_ui_state


def percentile(values: list[float], q: float) -> float:
  ordered = sorted(values)
  position = (len(ordered) - 1) * q / 100.0
  low = math.floor(position)
  high = math.ceil(position)
  if low == high:
    return ordered[low]
  return ordered[low] * (high - position) + ordered[high] * (position - low)


def main() -> None:
  parser = argparse.ArgumentParser(description='Synthetic comma-to-Mac accelerator link test')
  parser.add_argument('host', help='Mac IPv6 link-local address, including %usb0 scope')
  parser.add_argument('--port', type=int, default=8066)
  parser.add_argument('--frames', type=int, default=200)
  parser.add_argument('--warmup-frames', type=int, default=20)
  parser.add_argument('--qualification-frames', type=int, default=20)
  parser.add_argument('--frequency', type=float, default=20.0)
  parser.add_argument('--deadline-ms', type=float, default=50.0)
  parser.add_argument('--qualification-timeout-ms', type=float, default=500.0)
  parser.add_argument('--expected-output-floats', type=int, default=2576)
  parser.add_argument('--expected-backend', default='METAL')
  parser.add_argument('--expected-checkpoint')
  parser.add_argument('--expected-model-sha256')
  parser.add_argument('--auth-key-file', type=Path)
  parser.add_argument('--compression', choices=('zstd-1',))
  parser.add_argument('--synthetic-random-prefix-bytes', type=int, default=0,
                      help='non-sensitive incompressible prefix for realistic transport tests')
  parser.add_argument('--publish-ui-state', action='store_true',
                      help='publish Chestnut-style Mac worker state on a comma checkout')
  parser.add_argument('--realtime', action='store_true',
                      help='use the same realtime CPU/priority class as modeld on a comma')
  args = parser.parse_args()
  if (args.frames < 1 or args.warmup_frames < args.qualification_frames or args.qualification_frames < 1 or args.frequency <= 0 or
      args.deadline_ms <= 0 or args.qualification_timeout_ms < args.deadline_ms or args.expected_output_floats < 1):
    parser.error('frames, frequency, and deadline must be positive; warmup frames cannot be negative')
  if not 0 <= args.synthetic_random_prefix_bytes <= WARPED_BYTES:
    parser.error('synthetic random prefix must fit inside the warped payload')
  if args.realtime:
    from openpilot.common.realtime import config_realtime_process
    config_realtime_process(7, 54)

  policy = POLICY_INPUTS.pack(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.1, 0.1)
  random_prefix = random.Random(1).randbytes(args.synthetic_random_prefix_bytes)
  warped = random_prefix + bytes(WARPED_BYTES - len(random_prefix))
  period = 1.0 / args.frequency
  latencies: list[float] = []
  inference_times: list[float] = []
  prepare_times: list[float] = []
  send_times: list[float] = []
  receive_times: list[float] = []
  validate_times: list[float] = []
  attempted = 0

  client = AcceleratorClient(
    args.host, args.port, deadline_ms=args.deadline_ms,
    auth_key=read_auth_key(args.auth_key_file),
    expected_checkpoint=args.expected_checkpoint,
    expected_model_sha256=args.expected_model_sha256,
    expected_output_floats=args.expected_output_floats,
    expected_backend=args.expected_backend,
    qualification_frames=args.qualification_frames,
    qualification_timeout_ms=args.qualification_timeout_ms,
    compression=args.compression,
  )
  ui_state = make_ui_state(args.publish_ui_state)
  ui_state.loading()
  try:
    identity = client.connect()
  except Exception:
    ui_state.failed()
    raise
  ui_state.ready()
  print(f'accelerator={identity.device_type} backend={identity.backend} checkpoint={identity.model_checkpoint}')
  failed = False
  try:
    next_frame = time.monotonic()
    for frame_id in range(args.warmup_frames + args.frames):
      now = time.monotonic()
      if now < next_frame:
        time.sleep(next_frame - now)
      capture_ns = time.monotonic_ns()
      attempted += int(frame_id >= args.warmup_frames)
      result = client.infer(warped, policy, frame_id=frame_id, capture_ns=capture_ns, reset=frame_id == 0)
      if frame_id >= args.warmup_frames:
        if client.fallback.state.value != 'active':
          raise RuntimeError('accelerator did not pass warm-up qualification')
        latencies.append(result.round_trip_ms)
        inference_times.append(result.inference_ms)
        prepare_times.append(result.prepare_ms)
        send_times.append(result.send_ms)
        receive_times.append(result.receive_ms)
        validate_times.append(result.validate_ms)
      next_frame += period
  except Exception as error:
    failed = True
    ui_state.failed()
    print(f'stopped_on_failure={type(error).__name__}: {error}')
  finally:
    client.close()
    if not failed:
      ui_state.disconnected()

  print(f'frames_completed={len(latencies)}/{args.frames} warmup_frames={args.warmup_frames} frequency={args.frequency:.1f}Hz')
  if latencies:
    round_trip_summary = f'round_trip_ms mean={statistics.fmean(latencies):.2f} p95={percentile(latencies, 95):.2f} '
    round_trip_summary += f'p99={percentile(latencies, 99):.2f} max={max(latencies):.2f}'
    inference_summary = f'inference_ms mean={statistics.fmean(inference_times):.2f} p95={percentile(inference_times, 95):.2f} '
    inference_summary += f'p99={percentile(inference_times, 99):.2f} max={max(inference_times):.2f}'
    print(round_trip_summary)
    print(inference_summary)
    stage_summary = f'client_stage_ms prepare={statistics.fmean(prepare_times):.2f} '
    stage_summary += f'send={statistics.fmean(send_times):.2f} receive={statistics.fmean(receive_times):.2f} '
    stage_summary += f'validate={statistics.fmean(validate_times):.2f}'
    print(stage_summary)
  print(f'deadline_misses={attempted - len(latencies)}/{attempted} deadline_ms={args.deadline_ms:.1f}')
  fallback_message = f'chestnut_style_fallback={client.fallback.state.value} frame={client.fallback.failure_frame}'
  fallback_message += f' reason={client.fallback.failure_reason}'
  print(fallback_message)
  if len(latencies) != args.frames:
    raise SystemExit(1)


if __name__ == '__main__':
  main()
