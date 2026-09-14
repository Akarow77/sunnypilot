#!/usr/bin/env python3

from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from accelerator_client import AcceleratorIdentity, InferenceResult
from fallback import FallbackLatch
from live_shadow import LiveShadowWorker, pack_policy_inputs
from transport import DEVICE_TYPE, POLICY_INPUTS, WARPED_BYTES, WARPED_SHAPE


class FakeUIState:
  def __init__(self):
    self.states = []

  def loading(self): self.states.append('loading')
  def ready(self): self.states.append('ready')
  def active(self): self.states.append('active')
  def failed(self): self.states.append('failed')


class FakeTensor:
  def __init__(self, value):
    self.value = value

  def numpy(self):
    return self.value


class FakeClient:
  instances = []

  def __init__(self, _host, _port, **kwargs):
    self.fallback = FallbackLatch(kwargs['deadline_ms'])
    self.closed = False
    self.__class__.instances.append(self)

  def connect(self):
    return AcceleratorIdentity(
      DEVICE_TYPE, 'COREML_ANE', 'big', 'a' * 64, {}, {'outputs': [1, 18452]},
      18452, WARPED_BYTES + POLICY_INPUTS.size, ('zstd-1',), 'float16',
      {'action': (10, 14)},
    )

  def infer(self, _warped, _policy, *, frame_id, capture_ns, reset=False):
    self.fallback.activate()
    output = np.zeros(18452, dtype='<f4')
    output[10] = 4.0
    return InferenceResult(frame_id, memoryview(output).cast('B'), 10.0, 5.0, 1.0, 1.0, 2.0, 1.0)

  def close(self):
    self.closed = True


def policy_inputs():
  return {
    'desire_pulse': np.zeros(8, dtype=np.float32),
    'traffic_convention': np.array([1.0, 0.0], dtype=np.float32),
    'action_t': np.array([0.2, 0.8], dtype=np.float32),
  }


class LiveShadowTest(unittest.TestCase):
  def setUp(self):
    FakeClient.instances.clear()

  def test_policy_input_layout(self):
    packed = pack_policy_inputs(policy_inputs(), 'desire_pulse')
    self.assertEqual(len(packed), POLICY_INPUTS.size)
    self.assertEqual(POLICY_INPUTS.unpack(packed)[8:], (1.0, 0.0, 0.20000000298023224, 0.800000011920929))

  def test_latest_frame_replaces_unsent_frame(self):
    ui = FakeUIState()
    worker = LiveShadowWorker('::1%usb0', ui_state=ui)
    warped = FakeTensor(np.zeros(WARPED_SHAPE, dtype=np.uint8))
    for frame_id in (1, 2):
      self.assertTrue(worker.enqueue(warped, camera_frame_id=frame_id, capture_ns=frame_id,
                                     v_ego=10.0, numpy_inputs=policy_inputs(), desire_key='desire_pulse'))
    self.assertEqual(worker.dropped, 1)
    self.assertEqual(worker.queue.get_nowait().camera_frame_id, 2)

  def test_worker_logs_shadow_without_publishing_control(self):
    ui = FakeUIState()
    with tempfile.TemporaryDirectory() as temp_dir:
      key_path = Path(temp_dir) / 'auth.key'
      key_path.write_bytes(b'k' * 32)
      worker = LiveShadowWorker('::1%usb0', auth_key_path=key_path, log_dir=Path(temp_dir),
                                qualification_frames=1, client_factory=FakeClient, ui_state=ui)
      worker.start()
      warped = FakeTensor(np.zeros(WARPED_SHAPE, dtype=np.uint8))
      self.assertTrue(worker.enqueue(warped, camera_frame_id=9, capture_ns=time.monotonic_ns(),
                                     v_ego=2.0, numpy_inputs=policy_inputs(), desire_key='desire_pulse'))
      worker.record_local_action(9, 0.5)
      deadline = time.monotonic() + 1.0
      while worker.completed < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
      worker.stop()

      self.assertEqual(worker.completed, 1)
      self.assertIsNone(worker.failed_reason)
      self.assertIn('active', ui.states)
      logs = list(Path(temp_dir).glob('live-*.jsonl'))
      self.assertEqual(len(logs), 1)
      line = logs[0].read_text().strip()
      self.assertIn('"cameraFrameId":9', line)
      self.assertIn('"bigDesiredCurvature":1.0', line)
      self.assertIn('"localDesiredCurvature":0.5', line)
      self.assertTrue(FakeClient.instances[0].closed)


if __name__ == '__main__':
  unittest.main()
