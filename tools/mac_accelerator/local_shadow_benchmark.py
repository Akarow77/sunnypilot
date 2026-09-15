#!/usr/bin/env python3
"""Mac-only loopback: real ANE server, independent shadow process, paced producer.

Uses a private temporary key and separate localhost listener, never ADB or vehicle
parameters. Synthetic pixels test plumbing/timing; not driving-model accuracy.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import tempfile
import time

import numpy as np

from shadow_ipc import ShadowPublisher
from summarize_live_shadow import summarize_log, summary
from transport import WARPED_SHAPE


class HostTensor:
  def __init__(self, pixels):
    self.pixels = pixels

  def numpy(self):
    return self.pixels


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--frames', type=int, default=300)
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--drop-at', type=int, default=-1)
  parser.add_argument('--stall-at', type=int, default=-1, help='pause producer for 700ms; expect failed worker')
  parser.add_argument('--disconnect-at', type=int, default=-1, help='stop server; expect failed worker')
  args = parser.parse_args()
  if args.frames < 1:
    parser.error('positive frame count required')
  root = Path(__file__).resolve().parents[2]
  script_dir = Path(__file__).resolve().parent
  args.output_dir.mkdir(parents=True, exist_ok=True)
  onnx_path = root / 'openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx'
  with onnx_path.open('rb') as source:
    model_sha = hashlib.file_digest(source, 'sha256').hexdigest()
  with tempfile.TemporaryDirectory(prefix='mac-shadow-bench-') as directory:
    key_path = Path(directory) / 'auth.key'
    key_path.write_bytes(secrets.token_bytes(32))
    os.chmod(key_path, 0o600)
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as reservation:
      reservation.bind(('::1', 0))
      port = reservation.getsockname()[1]
    server_log_path = args.output_dir / 'server.log'
    with server_log_path.open('w') as server_log:
      server = subprocess.Popen([
        str(root / '.coreml-venv/bin/python'), str(script_dir / 'coreml_inference_server.py'),
        '--model', str(script_dir / 'artifacts/big_driving.mlpackage'),
        '--metadata', str(root / 'openpilot/selfdrive/modeld/models/big_driving_supercombo_metadata.pkl'),
        '--onnx', str(onnx_path), '--host', '::1', '--port', str(port), '--auth-key-file', str(key_path),
      ], stdout=server_log, stderr=subprocess.STDOUT,
        env={**os.environ, 'PYTHONUNBUFFERED': '1', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'})
      publisher = None
      try:
        timeout = time.monotonic() + 90
        while 'listening on' not in server_log_path.read_text():
          if server.poll() is not None or time.monotonic() > timeout:
            raise RuntimeError(f'ANE server startup failed; inspect {server_log_path}')
          time.sleep(.2)
        publisher = ShadowPublisher({'host': '::1', 'port': port, 'auth_key_path': str(key_path),
          'expected_model_sha256': model_sha, 'publish_ui': False, 'log_dir': str(args.output_dir)})
        publisher.start()
        # Start-up import work is outside the paced interval, without connecting.
        time.sleep(1)
        pixels = np.random.default_rng(1).integers(0, 256, WARPED_SHAPE, dtype=np.uint8)
        tensor = HostTensor(pixels)
        producer_ms = []
        schedule = time.monotonic()
        frame_id = 0
        for index in range(args.frames):
          if index == args.stall_at:
            time.sleep(.7)
          if index == args.disconnect_at:
            server.terminate()
            server.wait(timeout=5)
          if index == args.drop_at:
            frame_id += 1
          delay = schedule - time.monotonic()
          if delay > 0:
            time.sleep(delay)
          now = time.monotonic_ns()
          publisher.capture(tensor, camera_frame_id=frame_id, extra_frame_id=frame_id,
            capture_ns=now - 40_000_000, extra_capture_ns=now - 40_000_000,
            model_started_ns=now - 30_000_000, local_published_ns=now,
            calibrated=True, v_ego=10., local_curvature=.01, policy=[0.] * 8 + [1., 0., .2, .8])
          producer_ms.append((time.monotonic_ns() - now) / 1e6)
          frame_id += 1
          schedule += .05
          if publisher.failed_reason:
            break
        time.sleep(.2)
        rows = [summarize_log(path) for path in sorted(args.output_dir.glob('live-*.jsonl'))]
        report = {'test': 'Mac-only synthetic loopback; no 3X or controls', 'sourceModelSHA256': model_sha,
          'producerSnapshotMs': summary(producer_ms), 'producerFailure': publisher.failed_reason,
          'logs': rows, 'requestedFrames': args.frames, 'injectedDropAt': args.drop_at,
          'injectedStallAt': args.stall_at, 'injectedDisconnectAt': args.disconnect_at,
          'input': 'fixed synthetic random pixels; simulated camera timestamps and local curvature',
          'sustainedDeadlinePass': bool(rows) and all(row['frames'] == args.frames and row['qualifiedFrames'] > 0
            and not row['completedFrameDeadlineMisses'] and not row['failure'] for row in rows)}
        (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)
        if publisher.failed_reason or not rows or any(row['failure'] for row in rows):
          raise SystemExit(1)
      finally:
        if publisher is not None:
          publisher.close()
        server.terminate()
        try:
          server.wait(timeout=5)
        except subprocess.TimeoutExpired:
          server.kill()
          server.wait(timeout=5)


if __name__ == '__main__':
  main()
