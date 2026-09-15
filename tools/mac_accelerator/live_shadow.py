#!/usr/bin/env python3
"""Independent Mac Big Model shadow process. Remote outputs are logged only."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from accelerator_client import AcceleratorClient
from accelerator_protocol import read_auth_key
from shadow_ipc import FrameMailbox
from transport import POLICY_INPUTS
from ui_state import make_ui_state

AUTH_KEY_PATH = '/data/mac_accelerator/auth.key'
LOG_DIR = '/data/media/0/mac_accelerator_shadow'


class JsonLog:
  def __init__(self, path: Path, max_bytes: int = 32 * 1024 * 1024):
    self.file = path.open('x', buffering=1)
    os.chmod(path, 0o600)
    self.bytes = 0
    self.max_bytes = max_bytes

  def write(self, entry: dict):
    line = json.dumps(entry, allow_nan=False, separators=(',', ':')) + '\n'
    self.bytes += len(line.encode())
    if self.bytes > self.max_bytes:
      raise OSError('shadow log size limit reached')
    self.file.write(line)

  def close(self):
    self.file.close()


class ShadowSession:
  """Testable per-frame state machine: qualified contiguous context or failure."""
  def __init__(self, client, identity, ui, log, *, deadline_ms=55.0, max_capture_age_ms=150.0,
               qualification_frames=20, transport_compression='zstd-1'):
    if not 0 < deadline_ms <= 500 or not math.isfinite(max_capture_age_ms) or max_capture_age_ms < deadline_ms:
      raise ValueError('invalid shadow timing limits')
    shapes = identity.input_shapes
    if (shapes.get('img') != [1, 12, 128, 256] or shapes.get('big_img') != [1, 12, 128, 256]
        or shapes.get('desire_pulse') != [1, 33, 8] or shapes.get('action_t') != [1, 2]
        or shapes.get('traffic_convention') != [1, 2] or shapes.get('features_buffer') != [1, 32, 32, 512]
        or identity.output_floats != 18452 or identity.output_shapes != {'outputs': [1, 18452]}
        or identity.output_slices.get('action') != (2062, 2066) or identity.frame_skip != 2):
      raise ValueError('incompatible Big Model temporal/input/output contract')
    self.client, self.identity, self.ui, self.log = client, identity, ui, log
    self.deadline_ms, self.max_capture_age_ms = deadline_ms, max_capture_age_ms
    self.required = max(qualification_frames, 66)
    self.sequence = 0
    self.epoch = 0
    self.context_frames = 0
    self.good_frames = 0
    self.previous = None
    self.active = False
    self.failed = False
    self.ui.loading()
    self.log.write({'event': 'session', 'identity': asdict(identity), 'requiredContextFrames': self.required,
                    'deadlineMs': deadline_ms, 'maxCaptureAgeMs': max_capture_age_ms,
                    'transportCompression': transport_compression, 'controlSource': 'local'})

  def process(self, frame: dict, pixels: bytes):
    if self.failed:
      raise RuntimeError('shadow failure is latched')
    now = time.monotonic_ns()
    capture = frame['capture_ns']
    age_ms = (now - capture) / 1e6
    if not 0 <= age_ms <= self.max_capture_age_ms:
      raise RuntimeError(f'stale/future input frame: age={age_ms:.2f}ms')
    if not frame['calibrated'] or abs(capture - frame['extra_capture_ns']) > 10_000_000:
      raise RuntimeError('uncalibrated or unsynchronized camera frame')
    phase = frame.get('capture_phase', 'post-publish')
    if phase == 'post-warp':
      timing_order = capture <= frame['model_started_ns'] <= frame['warp_ready_ns'] <= frame['copy_started_ns'] <= frame['queued_ns'] <= now
    elif phase == 'post-publish':
      timing_order = capture <= frame['model_started_ns'] <= frame['local_published_ns'] <= frame['copy_started_ns'] <= frame['queued_ns'] <= now
    else:
      raise RuntimeError(f'unknown capture phase: {phase}')
    if not timing_order:
      raise RuntimeError('invalid frame timestamp order')
    local_curvature = frame.get('local_curvature')
    if (not math.isfinite(frame['v_ego']) or frame['v_ego'] < 0
        or (local_curvature is not None and not math.isfinite(local_curvature))):
      raise RuntimeError('invalid local comparison values')
    frame_id = frame['camera_frame_id']
    reset = self.previous is None
    missing = 0
    if self.previous is not None:
      previous_id, previous_capture = self.previous
      if frame_id <= previous_id or capture <= previous_capture:
        raise RuntimeError('duplicate/backward camera frame')
      missing = max(0, frame_id - previous_id - 1)
      reset = bool(missing or not 25_000_000 <= capture - previous_capture <= 75_000_000)
    if reset:
      self.epoch += 1
      self.context_frames = self.good_frames = 0
      self.active = False
      self.ui.loading()
      self.log.write({'event': 'contextReset', 'epoch': self.epoch, 'cameraFrameId': frame_id, 'missingFrames': missing})
    policy_values = frame['policy']
    if len(policy_values) != 12 or not all(math.isfinite(v) for v in policy_values):
      raise RuntimeError('invalid policy inputs')
    deadline_ns = (frame['copy_started_ns'] + int(self.deadline_ms * 1e6)) if self.active else None
    result = self.client.infer(pixels, POLICY_INPUTS.pack(*policy_values), frame_id=self.sequence,
                               capture_ns=capture, reset=reset, deadline_ns=deadline_ns)
    finished_ns = time.monotonic_ns()
    copy_to_output_ms = (finished_ns - frame['copy_started_ns']) / 1e6
    capture_to_output_ms = (finished_ns - capture) / 1e6
    late = copy_to_output_ms > self.deadline_ms or capture_to_output_ms > self.max_capture_age_ms
    was_active = self.active
    self.context_frames += 1
    self.good_frames = self.good_frames + 1 if not late else 0
    qualified = self.context_frames >= self.required and self.good_frames >= self.required
    if qualified and not self.active:
      self.ui.active()
      self.active = True
    big_curvature = float(np.frombuffer(result.output, dtype='<f4')[2062] / max(1.0, frame['v_ego']) ** 2)
    comparable = local_curvature is not None and qualified and not late and frame['v_ego'] >= 5.0
    entry = {
      'event': 'frame', 'cameraFrameId': frame_id, 'extraFrameId': frame['extra_frame_id'],
      'sequence': self.sequence, 'epoch': self.epoch, 'captureNs': capture,
      'vEgo': frame['v_ego'], 'copyToOutputMs': copy_to_output_ms, 'captureToOutputMs': capture_to_output_ms,
      'capturePhase': phase,
      'localModelMs': ((frame['local_published_ns'] - frame['model_started_ns']) / 1e6
                       if 'local_published_ns' in frame else None),
      'warpReadyMs': ((frame['warp_ready_ns'] - frame['model_started_ns']) / 1e6
                      if 'warp_ready_ns' in frame else None),
      'readbackMs': frame['readback_ms'], 'snapshotMs': (frame['queued_ns'] - frame['copy_started_ns']) / 1e6,
      'queueWaitMs': (now - frame['queued_ns']) / 1e6,
      'roundTripMs': result.round_trip_ms, 'inferenceMs': result.inference_ms,
      'prepareMs': result.prepare_ms, 'sendMs': result.send_ms,
      'receiveMs': result.receive_ms, 'validateMs': result.validate_ms,
      'deadlineMiss': late, 'qualified': qualified, 'comparable': comparable,
      'localDesiredCurvature': local_curvature, 'bigRawDesiredCurvature': big_curvature,
      'curvatureDifference': big_curvature - local_curvature if comparable else None,
      'missingFrames': missing, 'producerDrops': frame.get('producer_busy_drops', 0),
    }
    self.log.write(entry)
    self.ui.heartbeat()
    self.sequence += 1
    self.previous = frame_id, capture
    if late and was_active:
      raise RuntimeError(f'shadow deadline missed: copy={copy_to_output_ms:.2f}ms capture={capture_to_output_ms:.2f}ms')
    return entry

  def fail(self, reason: str, frame_id=None):
    self.failed = True
    try:
      self.log.write({'event': 'failure', 'error': reason, 'cameraFrameId': frame_id, 'completed': self.sequence})
    except Exception:
      pass
    try:
      self.ui.failed()
    finally:
      self.client.close()


def run(mailbox: FrameMailbox, config: dict, parent_pid: int):
  ui = make_ui_state(config.get('publish_ui', True))
  log = client = session = None
  last_sequence = 0
  last_frame_time = time.monotonic()
  current_frame = None
  try:
    ui.loading()
    directory = Path(config.get('log_dir', LOG_DIR))
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    used = sum(p.stat().st_size for p in directory.glob('live-*.jsonl'))
    if used >= 256 * 1024 * 1024:
      raise OSError('shadow log directory limit reached; archive logs before restarting')
    log = JsonLog(directory / f'live-{time.monotonic_ns()}.jsonl')
    while os.getppid() == parent_pid:
      packet = mailbox.receive(last_sequence)
      if packet is None:
        ui.heartbeat()
        if session is not None and time.monotonic() - last_frame_time > 0.5:
          raise TimeoutError('no camera frames for 500ms')
        # Calibration and camera-state publication can legitimately take longer
        # than a fixed startup window. Stay loading without connecting until the
        # first eligible frame; heartbeat expiry still detects worker death.
        time.sleep(0.002)
        continue
      last_sequence, current_frame, pixels = packet
      last_frame_time = time.monotonic()
      if session is None:
        key = read_auth_key(Path(config.get('auth_key_path', AUTH_KEY_PATH)))
        expected_hash = config['expected_model_sha256']
        if len(expected_hash) != 64 or any(c not in '0123456789abcdef' for c in expected_hash):
          raise ValueError('a pinned Big Model SHA-256 is required')
        compression = config.get('compression', 'zstd-1')
        client = AcceleratorClient(config['host'], config.get('port', 8066), auth_key=key,
                                   expected_model_sha256=expected_hash, expected_backend='COREML_ANE',
                                   expected_output_floats=18452, deadline_ms=config.get('deadline_ms', 55.0),
                                   qualification_frames=66, qualification_timeout_ms=500, compression=compression)
        identity = client.connect(timeout=2)
        session = ShadowSession(client, identity, ui, log, deadline_ms=config.get('deadline_ms', 55.0),
                                transport_compression=compression)
        if newer := mailbox.receive(last_sequence):
          last_sequence, current_frame, pixels = newer
      session.process(current_frame, pixels)
  except Exception as error:
    reason = f'{type(error).__name__}: {error}'
    if session is not None:
      session.fail(reason, current_frame.get('camera_frame_id') if current_frame else None)
    else:
      try:
        if log is not None:
          log.write({'event': 'failure', 'error': reason})
      except Exception:
        pass
      finally:
        ui.failed()
    return 1
  finally:
    if client is not None:
      client.close()
    if log is not None:
      log.close()
  ui.disconnected()
  return 0


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--mailbox', type=Path, required=True)
  parser.add_argument('--config', required=True)
  parser.add_argument('--parent-pid', type=int, required=True)
  args = parser.parse_args()
  mailbox = FrameMailbox(args.mailbox)
  try:
    raise SystemExit(run(mailbox, json.loads(args.config), args.parent_pid))
  finally:
    mailbox.close()


if __name__ == '__main__':
  main()
