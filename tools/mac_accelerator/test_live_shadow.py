from dataclasses import replace
import fcntl
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from accelerator_client import AcceleratorIdentity, InferenceResult
from live_shadow import JsonLog, ShadowSession, run
from shadow_ipc import FrameMailbox, ShadowPublisher
from transport import DEVICE_TYPE, REQUEST_BYTES, WARPED_BYTES, WARPED_SHAPE


def identity():
  return AcceleratorIdentity(DEVICE_TYPE, 'COREML_ANE', 'big', 'a' * 64,
    {'img': [1, 12, 128, 256], 'big_img': [1, 12, 128, 256], 'desire_pulse': [1, 33, 8],
     'traffic_convention': [1, 2], 'action_t': [1, 2], 'features_buffer': [1, 32, 32, 512]},
    {'outputs': [1, 18452]}, 18452, REQUEST_BYTES, ('zstd-1',), 'float16', {'action': (2062, 2066)}, 2)


def frame(frame_id, capture_ns):
  return {'camera_frame_id': frame_id, 'extra_frame_id': frame_id, 'capture_ns': capture_ns,
    'extra_capture_ns': capture_ns, 'calibrated': True, 'model_started_ns': capture_ns + 10_000_000,
    'local_published_ns': capture_ns + 25_000_000, 'copy_started_ns': capture_ns + 30_000_000,
    'queued_ns': capture_ns + 31_000_000, 'readback_ms': 0.1, 'v_ego': 10., 'local_curvature': .01,
    'policy': [0.] * 8 + [1., 0., .2, .8]}


class ShadowStateTests(unittest.TestCase):
  def setUp(self):
    self.ui, self.log, self.client = Mock(), Mock(), Mock()
    output = np.zeros(18452, dtype='<f4')
    output[2062] = 2.
    self.client.infer.return_value = InferenceResult(0, output.tobytes(), 25., 20., 1., 1., 1., 1.)
    self.session = ShadowSession(self.client, identity(), self.ui, self.log)
    self.now = 1_000_000_000

  def process(self, index, *, delay_ms=0, **overrides):
    self.now += 50_000_000
    sample = {**frame(index, self.now), **overrides}
    with patch('live_shadow.time.monotonic_ns', side_effect=[self.now + 40_000_000, self.now + int((40 + delay_ms) * 1e6)]):
      return self.session.process(sample, bytes(WARPED_BYTES))

  def test_loading_is_not_green_and_full_context_is_required(self):
    for i in range(65):
      self.assertFalse(self.process(i)['qualified'])
    self.ui.ready.assert_not_called()
    self.ui.active.assert_not_called()
    row = self.process(65)
    self.assertTrue(row['comparable'])
    self.assertAlmostEqual(row['curvatureDifference'], .01)
    self.ui.active.assert_called_once()

  def test_gap_resets_recurrence_and_qualification(self):
    for i in range(66):
      self.process(i)
    row = self.process(70)
    self.assertFalse(row['qualified'])
    self.assertEqual(row['missingFrames'], 4)
    self.assertTrue(self.client.infer.call_args.kwargs['reset'])
    self.assertEqual(row['epoch'], 2)

  def test_duplicate_frame_rejected_before_remote_inference(self):
    self.process(1)
    with self.assertRaisesRegex(RuntimeError, 'duplicate'):
      self.process(1)
    self.assertEqual(self.client.infer.call_count, 1)

  def test_stale_future_unsynchronized_uncalibrated_rejected(self):
    for overrides in ({'capture_ns': 1}, {'capture_ns': 99_000_000_000},
                      {'extra_capture_ns': 1}, {'calibrated': False}, {'v_ego': float('nan')}):
      with self.subTest(overrides=overrides), self.assertRaises(RuntimeError):
        self.process(1, **overrides)
    self.client.infer.assert_not_called()

  def test_failure_frame_is_logged_before_latching(self):
    for i in range(66):
      self.process(i)
    with self.assertRaisesRegex(RuntimeError, 'deadline'):
      self.process(66, delay_ms=70)
    self.assertTrue(self.log.write.call_args.args[0]['deadlineMiss'])
    self.session.fail('deadline', 66)
    self.assertTrue(self.session.failed)
    self.ui.failed.assert_called_once()
    self.client.close.assert_called_once()

  def test_log_failure_does_not_prevent_failed_ui_and_close(self):
    self.log.write.side_effect = OSError('disk full')
    self.session.fail('disk full')
    self.ui.failed.assert_called_once()
    self.client.close.assert_called_once()

  def test_context_contract_is_pinned(self):
    with self.assertRaisesRegex(ValueError, 'contract'):
      ShadowSession(self.client, replace(identity(), frame_skip=1), self.ui, self.log)

  def test_post_warp_frame_does_not_claim_local_comparison(self):
    sample = frame(1, self.now + 50_000_000)
    sample.pop('local_published_ns')
    sample.pop('local_curvature')
    sample['capture_phase'] = 'post-warp'
    sample['warp_ready_ns'] = sample['model_started_ns'] + 10_000_000
    sample['copy_started_ns'] = sample['warp_ready_ns'] + 1_000_000
    sample['queued_ns'] = sample['copy_started_ns'] + 1_000_000
    with patch('live_shadow.time.monotonic_ns', side_effect=[sample['queued_ns'] + 1_000_000,
                                                             sample['queued_ns'] + 30_000_000]):
      row = self.session.process(sample, bytes(WARPED_BYTES))
    self.assertEqual(row['capturePhase'], 'post-warp')
    self.assertIsNone(row['localDesiredCurvature'])
    self.assertFalse(row['comparable'])


class MailboxTests(unittest.TestCase):
  def test_invalid_publish_preserves_previous_packet(self):
    with tempfile.TemporaryDirectory() as directory:
      mailbox = FrameMailbox(Path(directory) / 'frames', create=True)
      self.addCleanup(mailbox.close)
      mailbox.publish(bytes(WARPED_BYTES), {'frame': 1})
      for metadata in ({'bad': float('nan')}, {'bad': 'x' * 5000}):
        with self.assertRaises(ValueError):
          mailbox.publish(b'x' * WARPED_BYTES, metadata)
        seq, meta, pixels = mailbox.receive(0)
        self.assertEqual((seq, meta['frame'], pixels), (1, 1, bytes(WARPED_BYTES)))

  def test_latest_frame_and_exclusion(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'frames'
      writer = FrameMailbox(path, create=True)
      reader = FrameMailbox(path)
      self.addCleanup(writer.close)
      self.addCleanup(reader.close)
      data = bytes(WARPED_BYTES)
      self.assertTrue(writer.publish(data, {'frame': 1}))
      self.assertTrue(writer.publish(data, {'frame': 2}))
      seq, meta, pixels = reader.receive(0)
      self.assertEqual((seq, meta['frame'], pixels), (2, 2, data))
      fcntl.flock(reader.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
      try:
        started = time.monotonic()
        self.assertFalse(writer.publish(data, {'frame': 3}))
        self.assertLess(time.monotonic() - started, .02)
      finally:
        fcntl.flock(reader.fd, fcntl.LOCK_UN)

  def test_readback_error_cannot_escape_and_is_latched(self):
    publisher = ShadowPublisher({})
    self.addCleanup(publisher.close)
    publisher.process = Mock()
    publisher.process.poll.return_value = None
    tensor = Mock()
    tensor.numpy.side_effect = RuntimeError('device failure')
    self.assertFalse(publisher.capture(tensor))
    self.assertFalse(publisher.capture(tensor))
    self.assertEqual(tensor.numpy.call_count, 1)
    self.assertIn('device failure', publisher.failed_reason)

  def test_dead_worker_never_reads_tensor(self):
    publisher = ShadowPublisher({})
    self.addCleanup(publisher.close)
    publisher.process = Mock()
    publisher.process.poll.return_value = 1
    tensor = Mock()
    self.assertFalse(publisher.capture(tensor))
    tensor.numpy.assert_not_called()
    self.assertIn('exited', publisher.failed_reason)

  def test_close_is_idempotent(self):
    publisher = ShadowPublisher({})
    publisher.close()
    publisher.close()

  def test_slow_readback_disables_future_copies(self):
    publisher = ShadowPublisher({}, copy_budget_ms=1)
    self.addCleanup(publisher.close)
    publisher.process = Mock()
    publisher.process.poll.return_value = None
    tensor = Mock()
    tensor.numpy.return_value = np.zeros(WARPED_SHAPE, dtype=np.uint8)
    with patch('shadow_ipc.time.monotonic_ns', side_effect=[0, 2_000_000, 2_100_000]):
      self.assertFalse(publisher.capture(tensor))
    self.assertIn('budget', publisher.failed_reason)
    consumer = FrameMailbox(publisher.path)
    self.addCleanup(consumer.close)
    with self.assertRaisesRegex(RuntimeError, 'readback exceeded copy budget'):
      consumer.receive(0)

  def test_mailbox_producer_error_is_bounded_and_terminal(self):
    with tempfile.TemporaryDirectory() as directory:
      mailbox = FrameMailbox(Path(directory) / 'frames', create=True)
      self.addCleanup(mailbox.close)
      self.assertTrue(mailbox.publish_error('broken'))
      with self.assertRaisesRegex(RuntimeError, 'producer failure: broken'):
        mailbox.receive(0)

  def test_modeld_starts_shadow_after_warp_before_local_policy(self):
    path = Path(__file__).resolve().parents[2] / 'openpilot/sunnypilot/modeld_v2/modeld.py'
    source = path.read_text()
    warp = source.index('warped = self.warp')
    callback = source.index('after_warp(warped)', warp)
    policy = source.index('raw_outputs = self.run_policy', callback)
    self.assertLess(warp, callback)
    self.assertLess(callback, policy)

  def test_first_device_readback_probe_is_parked_only(self):
    path = Path(__file__).resolve().parents[2] / 'openpilot/sunnypilot/modeld_v2/modeld.py'
    source = path.read_text()
    self.assertIn("parked_shadow_probe = v_ego < 0.5 and not sm['carControl'].latActive", source)
    self.assertIn('not calibrated or not should_probe', source)

  def test_next_parked_probe_disables_transport_compression(self):
    modeld = (Path(__file__).resolve().parents[2] / 'openpilot/sunnypilot/modeld_v2/modeld.py').read_text()
    worker = (Path(__file__).resolve().parent / 'live_shadow.py').read_text()
    self.assertIn("'compression': None", modeld)
    self.assertIn("compression = config.get('compression', 'zstd-1')", worker)
    self.assertIn('compression=compression', worker)


class WorkerLifecycleTests(unittest.TestCase):
  def test_no_first_frame_waits_without_connecting_until_parent_exits(self):
    with tempfile.TemporaryDirectory() as directory, patch('live_shadow.make_ui_state') as make_ui, \
         patch('live_shadow.os.getppid', side_effect=[1, 2]), patch('live_shadow.time.monotonic', return_value=0), \
         patch('live_shadow.time.sleep'), \
         patch('live_shadow.AcceleratorClient') as client:
      mailbox = Mock()
      mailbox.receive.return_value = None
      self.assertEqual(run(mailbox, {'log_dir': directory}, 1), 0)
      client.assert_not_called()
      make_ui.return_value.failed.assert_not_called()
      make_ui.return_value.disconnected.assert_called_once()
      self.assertEqual(next(Path(directory).glob('live-*.jsonl')).read_text(), '')

  def test_stalled_source_latches_and_closes_client(self):
    with tempfile.TemporaryDirectory() as directory, patch('live_shadow.make_ui_state'), \
         patch('live_shadow.os.getppid', return_value=1), patch('live_shadow.time.monotonic', side_effect=[0, .1, 1]), \
         patch('live_shadow.read_auth_key', return_value=b'x' * 32), \
         patch('live_shadow.AcceleratorClient') as client, patch('live_shadow.ShadowSession') as session:
      mailbox = Mock()
      mailbox.receive.side_effect = [(1, {'camera_frame_id': 1}, b''), None, None]
      self.assertEqual(run(mailbox, {'log_dir': directory, 'host': '::1', 'expected_model_sha256': 'a' * 64}, 1), 1)
      self.assertIn('500ms', session.return_value.fail.call_args.args[0])
      client.return_value.close.assert_called_once()

  def test_log_quota_rejects_without_deleting_existing_data(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'test.jsonl'
      log = JsonLog(path, max_bytes=20)
      self.addCleanup(log.close)
      log.write({'ok': 1})
      with self.assertRaises(OSError):
        log.write({'large': 'x' * 30})
      self.assertEqual(path.read_text(), '{"ok":1}\n')


if __name__ == '__main__':
  unittest.main()
