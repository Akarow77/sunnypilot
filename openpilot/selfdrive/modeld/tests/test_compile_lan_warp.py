"""CPU-buildable regression for dynamic calibration and serialized LAN warps."""
import pickle

import numpy as np
import pytest
from tinygrad import Device, Tensor

from openpilot.selfdrive.modeld.compile_lan_warp import compile_lan_warp, NV12Frame
from examples.openpilot.compile_warp import compile_warp
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info


@pytest.mark.parametrize('frame', [NV12Frame(64, 48, 80, 48, 24, 80 * 72),
                                 NV12Frame(1344, 760, *get_nv12_info(1344, 760)),
                                 NV12Frame(1928, 1208, *get_nv12_info(1928, 1208))])
def test_compiled_warp_roundtrip_and_dynamic_inputs(frame, tmp_path):
  # Exercise padded camera layouts, including both supported hardware sizes.
  artifact = compile_lan_warp(frame)
  assert artifact['input_specs']['frame'][0] == (frame.size,)
  assert artifact['input_specs']['M_inv'] == ((2, 3, 3), np.dtype(np.float32).str, 'NPY')
  artifact_path = tmp_path / f'lan_warp_{frame.width}x{frame.height}_tinygrad.pkl'
  with artifact_path.open('wb') as f:
    pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)
  with artifact_path.open('rb') as f:
    restored = pickle.load(f)
  assert restored['input_specs'] == artifact['input_specs']

  # Compare against upstream's packed two-camera warp and device transforms,
  # not just against the LAN closure that JIT captured.
  rng = np.random.default_rng(55)
  original_warp = compile_warp(frame, (512, 256), layout='yuv420', frames=2, benchmark_runs=1)['run']
  transforms = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
  transforms[0, 0, 0] = transforms[1, 0, 0] = frame.width / 512
  transforms[0, 1, 1] = transforms[1, 1, 1] = frame.height / 256
  matrix_tensor = Tensor(transforms, device='NPY').realize()
  for offset in (0, 11, -19):
    transforms[0, 0, 2] = offset
    transforms[1, 1, 2] = -offset
    narrow = Tensor(rng.integers(0, 256, frame.size, dtype=np.uint8), device=Device.DEFAULT).realize()
    wide = Tensor(rng.integers(0, 256, frame.size, dtype=np.uint8), device=Device.DEFAULT).realize()
    actual = restored['run'](frame=narrow, big_frame=wide, M_inv=matrix_tensor).numpy().copy()
    expected = original_warp(input_frame=Tensor.stack(narrow, wide).realize(),
                             M_inv=Tensor(transforms.copy(), device=Device.DEFAULT).realize()).numpy().copy()
    assert actual.shape == (2, 6, 128, 256) and actual.dtype == np.uint8
    np.testing.assert_array_equal(actual, expected)
