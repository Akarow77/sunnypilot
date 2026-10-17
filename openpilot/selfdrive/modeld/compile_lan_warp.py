"""Compile a two-camera native warp without concatenating full-resolution frames."""
import argparse

import numpy as np
from tinygrad import Device, Tensor
from examples.openpilot.compile_warp import make_frame_prepare, parse_frame
from examples.openpilot.helpers import allocate_inputs, compile_jit, dump_pickle


def compile_lan_warp(frame):
  prepare = make_frame_prepare(frame, 512, 256)
  specs = {'frame': ((frame.size,), np.dtype(np.uint8).str, Device.DEFAULT),
           'big_frame': ((frame.size,), np.dtype(np.uint8).str, Device.DEFAULT),
           'M_inv': ((2, 3, 3), np.dtype(np.float32).str, 'NPY')}

  def run(frame, big_frame, M_inv):
    # Upload both transforms once, before slicing. Slicing an NPY tensor before
    # its device copy can bake that slice into the JIT instead of tracking updates.
    transforms = M_inv.to(Device.DEFAULT).realize()
    return Tensor.stack(prepare(frame, transforms[0]), prepare(big_frame, transforms[1]))

  def inputs(seed):
    rng = np.random.default_rng(seed)
    def initialize(views):
      for name in ('frame', 'big_frame'):
        views[name][:] = rng.integers(0, 256, views[name].shape, dtype=np.uint8)
      views['M_inv'][:] = rng.standard_normal((2, 3, 3)) * 8
    return (), allocate_inputs(specs, initialize)

  return {'run': compile_jit(run, inputs, 1), 'input_specs': specs}


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--frame', type=parse_frame, required=True)
  parser.add_argument('--output', required=True)
  args = parser.parse_args()
  with open(args.output, 'wb') as f:
    dump_pickle(compile_lan_warp(args.frame), f)
