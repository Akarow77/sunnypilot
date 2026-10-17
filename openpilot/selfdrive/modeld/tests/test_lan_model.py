from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from openpilot.selfdrive.modeld.lan_model import LanModel, reset_local_model
from openpilot.selfdrive.modeld import lan_model
from openpilot.selfdrive.modeld.parse_model_outputs import Parser
from openpilot.selfdrive.modeld.gpu_on_lan import validate_metadata
from openpilot.selfdrive.modeld.tests.test_gpu_on_lan import metadata


def prepared_model():
  model = LanModel.__new__(LanModel)
  model.prev_desire = np.zeros(8, np.float32)
  model.timings = {}
  model.parser = Parser()
  model.prepare = Mock(return_value=np.zeros((2, 6, 128, 256), np.uint8))
  _, slices, size = validate_metadata(metadata())
  client = SimpleNamespace(output_slices=slices, infer=Mock(side_effect=lambda *args: np.zeros(size, np.float32)))
  return model, client


def run(model, client, desire):
  return model.run(client, None, None, np.eye(3), np.eye(3), desire, np.array([1, 0]), np.array([.2, .3]))


def test_native_outputs_and_independent_desire_history():
  model, client = prepared_model()
  desire = np.eye(8, dtype=np.float32)[3]
  result = run(model, client, desire)
  assert result['plan'].shape == (1, 33, 15)
  assert result['lead'].shape == (1, 3, 6, 4)
  np.testing.assert_array_equal(client.infer.call_args.args[1], desire)
  run(model, client, desire)
  assert not client.infer.call_args.args[1].any()
  assert desire[3] == 1  # caller's input was not mutated
  run(model, client, np.eye(8, dtype=np.float32)[0])
  assert not client.infer.call_args.args[1].any()


def test_remote_failure_does_not_commit_desire_history():
  model, client = prepared_model()
  client.infer.side_effect = TimeoutError('offline')
  with pytest.raises(TimeoutError):
    run(model, client, np.eye(8, dtype=np.float32)[3])
  assert not model.prev_desire.any()


def test_nonfinite_parsed_outputs_fail_closed():
  model, client = prepared_model()
  model.parser = Mock(parse_outputs=Mock(return_value={'plan': np.array([np.nan])}))
  with pytest.raises(ValueError, match='not finite'):
    run(model, client, np.eye(8, dtype=np.float32)[3])
  assert not model.prev_desire.any()


def test_raw_prediction_copy_precedes_parser_mutation(monkeypatch):
  monkeypatch.setattr(lan_model, 'SEND_RAW_PRED', '1')
  model, client = prepared_model()
  _, _, size = validate_metadata(metadata())
  original = np.arange(size, dtype=np.float32)
  raw = original.copy()
  client.infer = Mock(return_value=raw)

  def parse(outputs):
    for value in outputs.values():
      value[:] = -1
    return {'plan': np.zeros((1, 33, 15), np.float32)}

  model.parser = Mock(parse_outputs=Mock(side_effect=parse))
  result = run(model, client, np.zeros(8))
  np.testing.assert_array_equal(result['raw_pred'], original)
  assert not np.shares_memory(result['raw_pred'], raw)
  assert np.all(raw == -1)


def test_no_raw_prediction_copy_when_disabled(monkeypatch):
  monkeypatch.setattr(lan_model, 'SEND_RAW_PRED', None)
  model, client = prepared_model()
  _, _, size = validate_metadata(metadata())

  class NoCopyArray:
    def __init__(self):
      self.data = np.zeros(size, np.float32)

    def __getitem__(self, key):
      return self.data[key]

    def copy(self):
      raise AssertionError('normal LAN parsing must not copy raw predictions')

  client.infer = Mock(return_value=NoCopyArray())
  assert 'raw_pred' not in run(model, client, np.zeros(8))


def test_fallback_resets_legacy_gpu_queues_not_only_numpy_inputs():
  queues = {name: Mock() for name in ('img_q', 'big_img_q', 'desire_q', 'feat_q', 'tfm', 'new_img')}
  numpy_inputs = {name: np.ones((2, 8), np.float32) for name in ('prev_feat', 'desire', 'prev_desired_curv')}
  model = SimpleNamespace(input_queues=queues, numpy_inputs=numpy_inputs, prev_desire=np.ones(8, np.float32))
  reset_local_model(model)
  for value in numpy_inputs.values():
    assert not value.any()
  for name, tensor in queues.items():
    if name.endswith('_q'):
      tensor.assign.assert_called_once_with(0)
      tensor.assign.return_value.realize.assert_called_once_with()
    else:
      tensor.assign.assert_not_called()
  assert not model.prev_desire.any()


def test_fallback_resets_stock_numpy_views_without_adapter():
  model = SimpleNamespace(npy={'prev_feat': np.ones((1, 16), np.float32)}, input_queues={},
                          prev_desire=np.ones(8, np.float32))
  reset_local_model(model)
  assert not model.npy['prev_feat'].any()
  assert not model.prev_desire.any()


@pytest.mark.parametrize('transform', [np.eye(2), np.full((3, 3), np.nan)])
def test_bad_transform_rejected_before_mapping(transform):
  model = LanModel.__new__(LanModel)
  model.transforms = np.zeros((2, 3, 3), np.float32)
  model._frame = Mock()
  with pytest.raises(ValueError, match='transform'):
    model.prepare(None, None, transform, np.eye(3))
  model._frame.assert_not_called()


def test_prepared_view_is_synchronized_readonly_and_retains_buffer():
  model = LanModel.__new__(LanModel)
  model.transforms = np.zeros((2, 3, 3), np.float32)
  model._transform_tensor = object()
  model._image_dtype = object()
  model.timings = {}
  model._frame = Mock()
  data = bytearray(2 * 6 * 128 * 256)
  buffer = Mock(as_memoryview=Mock(return_value=memoryview(data)))
  warped = SimpleNamespace(shape=(2, 6, 128, 256), dtype=model._image_dtype, _buffer=Mock(return_value=buffer))
  model._warp = Mock(return_value=warped)
  image = model.prepare(None, None, np.eye(3), np.eye(3))
  buffer.as_memoryview.assert_called_once_with(allow_zero_copy=True)
  assert model._output_buffer is buffer
  assert image.shape == warped.shape and image.dtype == np.uint8
  assert not image.flags.writeable
  data[0] = 42
  assert image.flat[0] == 42  # borrowed, not an untracked extra full-image copy


def test_rejects_short_camera_buffer_before_gpu_mapping():
  model = LanModel.__new__(LanModel)
  model.frame_size = 16
  model._Tensor = Mock()
  with pytest.raises(ValueError, match='smaller'):
    model._frame(np.zeros(8, np.uint8))
  model._Tensor.from_blob.assert_not_called()

