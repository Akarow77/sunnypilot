#!/usr/bin/env python3
"""Non-controlling live shadow worker for a comma 3X.

The modeld thread only copies the already-computed warp and performs a nonblocking
latest-frame enqueue. Network inference and logging happen on a daemon thread. No
remote output is ever published to cereal or returned to modeld.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
from queue import Empty, Full, Queue
import sys
import threading
import time
from typing import Any

import numpy as np

from accelerator_client import AcceleratorClient
from accelerator_protocol import read_auth_key
from fallback import AcceleratorState
from transport import POLICY_INPUTS, WARPED_BYTES, WARPED_SHAPE
from ui_state import make_ui_state


AUTH_KEY_PATH = Path('/data/mac_accelerator/auth.key')
LOG_DIR = Path('/data/media/0/mac_accelerator_shadow')


@dataclass(frozen=True)
class ShadowFrame:
  camera_frame_id: int
  capture_ns: int
  warp_started_ns: int
  queued_ns: int
  v_ego: float
  warped: bytes
  policy: bytes


def pack_policy_inputs(numpy_inputs: dict[str, np.ndarray], desire_key: str) -> bytes:
  desire = np.asarray(numpy_inputs[desire_key], dtype=np.float32).reshape(-1)
  traffic = np.asarray(numpy_inputs['traffic_convention'], dtype=np.float32).reshape(-1)
  action_t = np.asarray(numpy_inputs['action_t'], dtype=np.float32).reshape(-1)
  if desire.size != 8 or traffic.size != 2 or action_t.size != 2:
    raise ValueError(f'incompatible policy inputs: desire={desire.size} traffic={traffic.size} action_t={action_t.size}')
  return POLICY_INPUTS.pack(*(desire.tolist() + traffic.tolist() + action_t.tolist()))


class LiveShadowWorker:
  """Latest-frame-only remote inference worker with latched failure behavior."""

  def __init__(self, host: str, *, port: int = 8066, deadline_ms: float = 55.0,
               expected_model_sha256: str | None = None, auth_key_path: Path = AUTH_KEY_PATH,
               log_dir: Path = LOG_DIR, qualification_frames: int = 20,
               client_factory=AcceleratorClient, ui_state=None):
    self.host = host
    self.port = port
    self.deadline_ms = deadline_ms
    self.expected_model_sha256 = expected_model_sha256 or None
    self.auth_key_path = auth_key_path
    self.log_dir = log_dir
    self.qualification_frames = qualification_frames
    self.client_factory = client_factory
    self.ui_state = ui_state or make_ui_state(True)
    self.queue: Queue[ShadowFrame] = Queue(maxsize=1)
    self.stop_event = threading.Event()
    self.local_actions: OrderedDict[int, float] = OrderedDict()
    self.local_lock = threading.Lock()
    self.thread: threading.Thread | None = None
    self.failed_reason: str | None = None
    self.enqueued = 0
    self.dropped = 0
    self.completed = 0

  @classmethod
  def from_params(cls, params):
    if not params.get_bool('MacAcceleratorShadowEnabled'):
      return None
    host = params.get('MacAcceleratorHost')
    if not host:
      worker = cls('')
      worker.fail('MacAcceleratorHost is not configured')
      return worker
    return cls(
      host,
      port=params.get('MacAcceleratorPort', return_default=True),
      deadline_ms=params.get('MacAcceleratorDeadlineMs', return_default=True),
      expected_model_sha256=params.get('MacAcceleratorExpectedModelSHA256'),
    )

  def start(self) -> None:
    if self.failed_reason is not None or self.thread is not None:
      return
    self.ui_state.loading()
    self.thread = threading.Thread(target=self._run, name='mac-accelerator-shadow', daemon=True)
    self.thread.start()

  def stop(self) -> None:
    self.stop_event.set()
    if self.thread is not None:
      self.thread.join(timeout=2.0)

  def fail(self, reason: str) -> None:
    if self.failed_reason is None:
      self.failed_reason = reason
      print(f'Mac accelerator live shadow failed: {reason}', file=sys.stderr, flush=True)
      self.ui_state.failed()

  def enqueue(self, warped_tensor: Any, *, camera_frame_id: int, capture_ns: int,
              v_ego: float, numpy_inputs: dict[str, np.ndarray], desire_key: str) -> bool:
    if self.failed_reason is not None:
      return False
    warp_started_ns = time.monotonic_ns()
    try:
      warped_array = np.asarray(warped_tensor.numpy())
      if warped_array.shape != WARPED_SHAPE or warped_array.dtype != np.uint8 or warped_array.nbytes != WARPED_BYTES:
        raise ValueError(f'invalid live warp: shape={warped_array.shape} dtype={warped_array.dtype}')
      frame = ShadowFrame(
        camera_frame_id=camera_frame_id,
        capture_ns=capture_ns,
        warp_started_ns=warp_started_ns,
        queued_ns=time.monotonic_ns(),
        v_ego=v_ego,
        warped=warped_array.tobytes(),
        policy=pack_policy_inputs(numpy_inputs, desire_key),
      )
    except Exception as error:
      self.fail(f'live warp capture failed: {type(error).__name__}: {error}')
      return False

    try:
      self.queue.put_nowait(frame)
    except Full:
      try:
        self.queue.get_nowait()
      except Empty:
        pass
      self.dropped += 1
      try:
        self.queue.put_nowait(frame)
      except Full:
        self.dropped += 1
        return False
    self.enqueued += 1
    return True

  def record_local_action(self, camera_frame_id: int, desired_curvature: float) -> None:
    with self.local_lock:
      self.local_actions[camera_frame_id] = desired_curvature
      while len(self.local_actions) > 128:
        self.local_actions.popitem(last=False)

  def _take_local_action(self, camera_frame_id: int) -> float | None:
    with self.local_lock:
      return self.local_actions.pop(camera_frame_id, None)

  def _make_client(self):
    return self.client_factory(
      self.host,
      self.port,
      deadline_ms=self.deadline_ms,
      auth_key=read_auth_key(self.auth_key_path),
      expected_model_sha256=self.expected_model_sha256,
      expected_output_floats=18452,
      expected_backend='COREML_ANE',
      # LiveShadowWorker qualifies the complete warp-to-output path below.
      qualification_frames=1,
      qualification_timeout_ms=500.0,
      compression='zstd-1',
    )

  @staticmethod
  def _big_curvature(output: bytes | memoryview, output_slices: dict[str, tuple[int, int]], v_ego: float) -> float | None:
    action_slice = output_slices.get('action')
    if action_slice is None or action_slice[1] - action_slice[0] < 1:
      return None
    values = np.frombuffer(output, dtype='<f4')
    return float(values[action_slice[0]] / max(1.0, v_ego) ** 2)

  def _run(self) -> None:
    client = None
    log_file = None
    try:
      client = self._make_client()
      identity = client.connect()
      if 'action' not in identity.output_slices:
        raise RuntimeError('accelerator identity does not publish the Big Model action slice')
      self.ui_state.ready()
      self.log_dir.mkdir(parents=True, exist_ok=True)
      log_path = self.log_dir / f'live-{time.monotonic_ns()}.jsonl'
      log_file = log_path.open('a', buffering=1)
      sequence = 0
      qualified = 0
      active_published = False
      while not self.stop_event.is_set():
        try:
          frame = self.queue.get(timeout=0.25)
        except Empty:
          continue
        infer_started_ns = time.monotonic_ns()
        result = client.infer(frame.warped, frame.policy, frame_id=sequence,
                              capture_ns=frame.capture_ns, reset=sequence == 0)
        completed_ns = time.monotonic_ns()
        warp_to_output_ms = (completed_ns - frame.warp_started_ns) / 1e6
        if warp_to_output_ms > self.deadline_ms:
          client.fallback.fail(sequence, f'warp-to-output deadline missed: {warp_to_output_ms:.2f} ms')
          raise RuntimeError(client.fallback.failure_reason)
        qualified += 1
        if (not active_published and client.fallback.state is AcceleratorState.ACTIVE and
            qualified >= self.qualification_frames):
          self.ui_state.active()
          active_published = True
        big_curvature = self._big_curvature(result.output, identity.output_slices, frame.v_ego)
        local_curvature = self._take_local_action(frame.camera_frame_id)
        entry = {
          'cameraFrameId': frame.camera_frame_id,
          'sequence': sequence,
          'captureNs': frame.capture_ns,
          'vEgo': frame.v_ego,
          'warpCopyMs': (frame.queued_ns - frame.warp_started_ns) / 1e6,
          'queueWaitMs': (infer_started_ns - frame.queued_ns) / 1e6,
          'roundTripMs': result.round_trip_ms,
          'inferenceMs': result.inference_ms,
          'warpToOutputMs': warp_to_output_ms,
          'localDesiredCurvature': local_curvature,
          'bigDesiredCurvature': big_curvature,
          'curvatureDifference': (big_curvature - local_curvature
                                  if big_curvature is not None and local_curvature is not None else None),
          'producerDrops': self.dropped,
        }
        log_file.write(json.dumps(entry, separators=(',', ':')) + '\n')
        sequence += 1
        self.completed += 1
    except Exception as error:
      if log_file is not None:
        log_file.write(json.dumps({'error': f'{type(error).__name__}: {error}',
                                   'completed': self.completed, 'producerDrops': self.dropped},
                                  separators=(',', ':')) + '\n')
      self.fail(f'{type(error).__name__}: {error}')
    finally:
      if client is not None:
        client.close()
      if log_file is not None:
        log_file.close()
