#!/usr/bin/env python3
"""Runtime adapter from comma VisionIPC buffers to remote warped model tensors."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from tinygrad.tensor import Tensor

from openpilot.selfdrive.modeld.compile_modeld import NV12Frame, nv12_copy_size
from openpilot.selfdrive.modeld.helpers import load_oob
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

try:
  from .transport import WARPED_BYTES, WARPED_SHAPE
except ImportError:
  from transport import WARPED_BYTES, WARPED_SHAPE


class WarpRuntime:
  def __init__(self, artifact_path: Path):
    with artifact_path.open('rb') as artifact_file:
      artifact = load_oob(artifact_file)
    self.camera_size = tuple(artifact['camera_size'])
    self.model_size = tuple(artifact['model_size'])
    self.device = str(artifact['input_device'])
    self.warp = artifact['warp']
    width, height = self.camera_size
    nv12 = NV12Frame(width, height, *get_nv12_info(width, height))
    self.frame_size = nv12_copy_size(nv12.stride, nv12.y_height, nv12.uv_height)
    self.npy = {
      'tfm': np.zeros((3, 3), dtype=np.float32),
      'big_tfm': np.zeros((3, 3), dtype=np.float32),
    }
    self.queues = {name: Tensor(value, device='NPY').realize() for name, value in self.npy.items()}
    self._blob_cache: dict[tuple[str, int], Tensor] = {}

  def _frame_tensor(self, name: str, buffer) -> Tensor:
    if isinstance(buffer, (bytes, bytearray, memoryview, np.ndarray)):
      view = np.frombuffer(buffer, dtype=np.uint8, count=self.frame_size)
      return Tensor(view, device=self.device).contiguous().realize()
    data = buffer.data if hasattr(buffer, 'data') else buffer
    view = np.frombuffer(data, dtype=np.uint8, count=self.frame_size)
    key = (name, view.ctypes.data)
    if key not in self._blob_cache:
      self._blob_cache[key] = Tensor.from_blob(view.ctypes.data, (self.frame_size,), dtype='uint8', device=self.device)
    return self._blob_cache[key]

  def run(self, narrow_buffer, wide_buffer, narrow_transform: np.ndarray,
          wide_transform: np.ndarray) -> tuple[bytes, float]:
    self.npy['tfm'][:] = np.asarray(narrow_transform, dtype=np.float32).reshape(3, 3)
    self.npy['big_tfm'][:] = np.asarray(wide_transform, dtype=np.float32).reshape(3, 3)
    started = time.monotonic_ns()
    output = self.warp(
      tfm=self.queues['tfm'],
      big_tfm=self.queues['big_tfm'],
      frame=self._frame_tensor('frame', narrow_buffer),
      big_frame=self._frame_tensor('big_frame', wide_buffer),
    ).numpy().astype(np.uint8, copy=False)
    elapsed_ms = (time.monotonic_ns() - started) / 1e6
    if output.shape != WARPED_SHAPE or output.nbytes != WARPED_BYTES:
      raise RuntimeError(f'invalid warp output: shape={output.shape} bytes={output.nbytes}')
    return output.tobytes(), elapsed_ms
