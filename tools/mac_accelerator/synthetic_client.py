#!/usr/bin/env python3
"""Dependency-free 20 Hz synthetic client suitable for running on a comma 3X."""

from __future__ import annotations

import argparse
import math
import secrets
import socket
import statistics
import time

from fallback import FallbackLatch
from transport import (FLAG_RESET, POLICY_INPUTS, REQUEST, RESPONSE, TIMINGS,
                       WARPED_BYTES, recv_message, send_message)


def percentile(values: list[float], q: float) -> float:
  ordered = sorted(values)
  position = (len(ordered) - 1) * q / 100.0
  low = math.floor(position)
  high = math.ceil(position)
  if low == high:
    return ordered[low]
  return ordered[low] * (high - position) + ordered[high] * (position - low)


def main() -> None:
  parser = argparse.ArgumentParser(description="Synthetic comma-to-Mac accelerator link test")
  parser.add_argument('host', help="Mac IPv6 link-local address, including %usb0 scope")
  parser.add_argument('--port', type=int, default=8066)
  parser.add_argument('--frames', type=int, default=200)
  parser.add_argument('--warmup-frames', type=int, default=5)
  parser.add_argument('--frequency', type=float, default=20.0)
  parser.add_argument('--deadline-ms', type=float, default=50.0)
  parser.add_argument('--expected-output-floats', type=int, default=2576)
  args = parser.parse_args()
  if (args.frames < 1 or args.warmup_frames < 0 or args.frequency <= 0 or
      args.deadline_ms <= 0 or args.expected_output_floats < 1):
    parser.error('frames, frequency, and deadline must be positive; warmup frames cannot be negative')

  policy = POLICY_INPUTS.pack(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.1, 0.1)
  payload = bytes(WARPED_BYTES) + policy
  session_id = secrets.randbits(64)
  period = 1.0 / args.frequency
  latencies: list[float] = []
  inference_times: list[float] = []
  deadline_misses = 0
  fallback = FallbackLatch(args.deadline_ms)

  with socket.create_connection((args.host, args.port), timeout=5.0) as conn:
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 * 1024 * 1024)
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    conn.settimeout(max(5.0, args.deadline_ms / 1000.0 * 4))
    next_frame = time.monotonic()
    for frame_id in range(args.warmup_frames + args.frames):
      now = time.monotonic()
      if now < next_frame:
        time.sleep(next_frame - now)
      capture_ns = time.monotonic_ns()
      flags = FLAG_RESET if frame_id == 0 else 0
      send_message(conn, REQUEST, flags, session_id, frame_id, capture_ns, payload)
      response_flags, response_session, response_frame, response_capture, response = recv_message(conn, RESPONSE)
      completed_ns = time.monotonic_ns()
      if (response_session, response_frame, response_capture) != (session_id, frame_id, capture_ns):
        raise RuntimeError('response identity mismatch')
      if response_flags != flags & FLAG_RESET:
        raise RuntimeError('reset acknowledgement mismatch')
      expected_response_bytes = TIMINGS.size + args.expected_output_floats * 4
      if len(response) != expected_response_bytes:
        raise RuntimeError(f'invalid response length: {len(response)}')
      _, inference_start_ns, inference_end_ns = TIMINGS.unpack_from(response)
      latency_ms = (completed_ns - capture_ns) / 1e6
      output_valid = all(math.isfinite(value) for value in memoryview(response)[TIMINGS.size:].cast('f'))
      if not output_valid:
        raise RuntimeError(f'non-finite model output at frame {frame_id}')
      if frame_id >= args.warmup_frames:
        if frame_id == args.warmup_frames:
          fallback.activate()
        latencies.append(latency_ms)
        inference_times.append((inference_end_ns - inference_start_ns) / 1e6)
        deadline_misses += latency_ms > args.deadline_ms
        fallback.observe(frame_id, latency_ms, output_valid)
      next_frame += period

  print(f'frames={args.frames} warmup_frames={args.warmup_frames} frequency={args.frequency:.1f}Hz')
  round_trip_summary = f'round_trip_ms mean={statistics.fmean(latencies):.2f} p95={percentile(latencies, 95):.2f} '
  round_trip_summary += f'p99={percentile(latencies, 99):.2f} max={max(latencies):.2f}'
  inference_summary = f'inference_ms mean={statistics.fmean(inference_times):.2f} p95={percentile(inference_times, 95):.2f} '
  inference_summary += f'p99={percentile(inference_times, 99):.2f} max={max(inference_times):.2f}'
  print(round_trip_summary)
  print(inference_summary)
  print(f'deadline_misses={deadline_misses}/{args.frames} deadline_ms={args.deadline_ms:.1f}')
  print(f'chestnut_style_fallback={fallback.state.value} frame={fallback.failure_frame} reason={fallback.failure_reason}')


if __name__ == '__main__':
  main()
