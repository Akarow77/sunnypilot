"""Native LAN preprocessing/decoding, independent of the selected local driving model."""
import os
import pickle
import time

import numpy as np

from openpilot.selfdrive.modeld.helpers import MODELS_DIR
from openpilot.selfdrive.modeld.parse_model_outputs import Parser
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

SEND_RAW_PRED = os.getenv('SEND_RAW_PRED')


class LanModel:
  def __init__(self, cam_w: int, cam_h: int):
    from tinygrad import Tensor, dtypes
    self._Tensor = Tensor
    self._image_dtype = dtypes.uint8
    self.frame_size = get_nv12_info(cam_w, cam_h)[3]
    with open(MODELS_DIR / f'lan_warp_{cam_w}x{cam_h}_tinygrad.pkl', 'rb') as f:
      artifact = pickle.load(f)
    self._warp = artifact['run']
    self.device = artifact['input_specs']['frame'][2]
    self.transforms = np.zeros((2, 3, 3), dtype=np.float32)
    self._transform_tensor = Tensor(self.transforms, device='NPY').realize()
    self._buffers = {}
    self.prev_desire = np.zeros(8, dtype=np.float32)
    self.parser = Parser()
    self.timings = {}

  def _frame(self, frame):
    data = np.frombuffer(frame.data, dtype=np.uint8) if hasattr(frame, 'data') else np.asarray(frame, dtype=np.uint8)
    if data.size < self.frame_size:
      raise ValueError('camera buffer is smaller than its NV12 layout')
    ptr = data.ctypes.data
    if ptr not in self._buffers:
      # A finite camera ring; cap retained mappings if a producer changes buffers.
      if len(self._buffers) >= 64:
        self._buffers.clear()
      self._buffers[ptr] = self._Tensor.from_blob(ptr, (self.frame_size,), dtype='uint8', device=self.device)
    return self._buffers[ptr]

  def prepare(self, narrow, wide, narrow_transform, wide_transform):
    """Return a read-only image view, valid until the next synchronous prepare call."""
    started = time.perf_counter()
    for i, transform in enumerate((narrow_transform, wide_transform)):
      if np.shape(transform) != (3, 3) or not np.all(np.isfinite(transform)):
        raise ValueError('invalid LAN camera transform')
      self.transforms[i] = transform
    frame, big_frame = self._frame(narrow), self._frame(wide)
    mapped = time.perf_counter()
    self.timings['map_ms'] = (mapped - started) * 1000
    warped = self._warp(frame=frame, big_frame=big_frame, M_inv=self._transform_tensor)
    submitted = time.perf_counter()
    if warped.shape != (2, 6, 128, 256) or warped.dtype != self._image_dtype:
      raise ValueError('unexpected native LAN warp layout')
    # QCOM output is CPU-mapped already. Synchronize before reading, retain the
    # allocation and borrow a read-only view instead of copying via .numpy().
    # run() encodes/sends it synchronously before another warp can overwrite it.
    self._output_buffer = warped._buffer()
    image = np.frombuffer(self._output_buffer.as_memoryview(allow_zero_copy=True), dtype=np.uint8).reshape(warped.shape)
    image.flags.writeable = False
    finished = time.perf_counter()
    self.timings.update(warp_submit_ms=(submitted-mapped)*1000,
                        warp_readback_ms=(finished-submitted)*1000, prepare_ms=(finished-started)*1000)
    return image

  def run(self, client, narrow, wide, narrow_transform, wide_transform, desire, traffic_convention, action_t):
    image = self.prepare(narrow, wide, narrow_transform, wide_transform)
    desire = np.asarray(desire, dtype=np.float32).reshape(8).copy()
    desire[0] = 0
    pulse = np.where(desire - self.prev_desire > .99, desire, 0).astype(np.float32)
    started = time.perf_counter()
    raw = client.infer(image, pulse, traffic_convention, action_t)
    self.timings['remote_ms'] = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    # Parser applies softmax in place on some slices. Preserve original network
    # predictions only when requested by the native message writer.
    raw_pred = raw.copy() if SEND_RAW_PRED else None
    outputs = self.parser.parse_outputs({k: raw[np.newaxis, v] for k, v in client.output_slices.items()})
    if not all(np.all(np.isfinite(v)) for v in outputs.values()):
      raise ValueError('parsed LAN output is not finite')
    if raw_pred is not None:
      outputs['raw_pred'] = raw_pred
    self.prev_desire[:] = desire
    self.timings['parse_ms'] = (time.perf_counter() - started) * 1000
    return outputs


def reset_local_model(model):
  """Discard history accumulated before remote inference, including legacy GPU queues."""
  adapter = getattr(model, 'adapter', None)
  state = adapter if adapter is not None else model
  if adapter is not None:
    adapter.reset_warmup_buffers()
  else:
    if hasattr(model, 'packed_input'):
      model.packed_input[:] = 0
    for value in getattr(model, 'numpy_inputs', getattr(model, 'npy', {})).values():
      value[:] = 0
  # Native adapters reset state_pairs themselves. Legacy adapters only clear
  # host inputs; their recurrent GPU queues also need to be explicitly zeroed.
  state_names = getattr(model, 'state_pairs', {}) if adapter is None else {}
  for name, tensor in state.input_queues.items():
    if name.endswith('_q') or name in state_names:
      tensor.assign(0).realize()
  model.prev_desire[:] = 0
