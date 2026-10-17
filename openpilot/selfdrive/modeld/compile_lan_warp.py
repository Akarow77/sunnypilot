"""Compile the LAN model's native two-camera 512x256 NV12 warp.

This uses the compiler shipped with this branch, independently of the selected
local model's resolution and recurrent queues. No full-frame concatenation is
needed: camera buffers remain separate device inputs.
"""
import argparse
import pickle

import numpy as np
from tinygrad import Device, Tensor
from tinygrad.engine.jit import TinyJit

from openpilot.selfdrive.modeld.compile_modeld import NV12Frame, make_frame_prepare, _parse_size
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info


def make_lan_warp(nv12):
  prepare = make_frame_prepare(nv12, 512, 256)

  def run(frame, big_frame, M_inv):
    # Copy the complete NPY input before slicing. Slicing on NPY can capture a
    # calibration-dependent constant instead of replaying updated transforms.
    transforms = M_inv.to(Device.DEFAULT).realize()
    return Tensor.stack(prepare(frame, transforms[0]), prepare(big_frame, transforms[1])).realize()

  return run


def compile_lan_warp(nv12):
  run = make_lan_warp(nv12)
  jit = TinyJit(run, prune=True)
  specs = {'frame': ((nv12.size,), np.dtype(np.uint8).str, Device.DEFAULT),
           'big_frame': ((nv12.size,), np.dtype(np.uint8).str, Device.DEFAULT),
           'M_inv': ((2, 3, 3), np.dtype(np.float32).str, 'NPY')}
  rng = np.random.default_rng(42)
  frames = [rng.integers(0, 256, nv12.size, dtype=np.uint8) for _ in range(2)]
  transforms = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
  transform_tensor = Tensor(transforms, device='NPY').realize()
  inputs = {'frame': Tensor(frames[0], device=Device.DEFAULT).realize(),
            'big_frame': Tensor(frames[1], device=Device.DEFAULT).realize(), 'M_inv': transform_tensor}

  for _ in range(3):  # eager execution, capture, and replay
    np.testing.assert_array_equal(jit(**inputs).numpy(), run(**inputs).numpy())

  # Validate the serialized artifact with the same NPY allocation updated in
  # place, exactly as modeld does. Check calibration and frame changes separately
  # so a stale transform cannot hide behind fresh image contents.
  loaded = pickle.loads(pickle.dumps(jit, protocol=pickle.HIGHEST_PROTOCOL))
  previous = loaded(**inputs).numpy().copy()
  for change in ('calibration', 'frame', 'both'):
    if change in ('calibration', 'both'):
      transforms[:, 0, 2] += 13
      transforms[:, 1, 2] += 7
    if change in ('frame', 'both'):
      for name in ('frame', 'big_frame'):
        inputs[name] = Tensor(rng.integers(0, 256, nv12.size, dtype=np.uint8), device=Device.DEFAULT).realize()
    expected = run(**inputs).numpy().copy()
    actual = loaded(**inputs).numpy().copy()
    np.testing.assert_array_equal(actual, expected, err_msg=f'LAN warp replay differs after {change} change')
    if np.array_equal(actual, previous):
      raise ValueError(f'LAN warp did not respond to {change} change')
    previous = actual

  return {'run': jit, 'input_specs': specs}


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--camera-resolution', type=_parse_size, required=True)
  parser.add_argument('--output', required=True)
  args = parser.parse_args()
  cam_w, cam_h = args.camera_resolution
  nv12 = NV12Frame(cam_w, cam_h, *get_nv12_info(cam_w, cam_h))
  artifact = compile_lan_warp(nv12)
  with open(args.output, 'wb') as f:
    pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)
