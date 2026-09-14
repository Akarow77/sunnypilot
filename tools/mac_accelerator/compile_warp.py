#!/usr/bin/env python3
"""Compile the camera warp that prepares remote-model tensors on the comma 3X."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
from tinygrad import dtypes
from tinygrad.device import Device
from tinygrad.engine.jit import TinyJit
from tinygrad.tensor import Tensor

from openpilot.selfdrive.modeld.compile_modeld import NV12Frame, make_warp, nv12_copy_size
from openpilot.selfdrive.modeld.helpers import dump_oob
from openpilot.sunnypilot.modeld_v2.compile_modeld import compile_jit
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info


WARP_INPUTS = ['tfm', 'big_tfm', 'frame', 'big_frame']


def parse_size(value: str) -> tuple[int, int]:
  width, height = value.lower().split('x')
  return int(width), int(height)


def make_queues(camera_size: tuple[int, int], device: str):
  npy = {
    'tfm': np.zeros((3, 3), dtype=np.float32),
    'big_tfm': np.zeros((3, 3), dtype=np.float32),
  }
  queues = {
    'tfm': Tensor(npy['tfm'], device='NPY').realize(),
    'big_tfm': Tensor(npy['big_tfm'], device='NPY').realize(),
  }
  return queues, npy


def make_random_frames(camera_size: tuple[int, int], device: str, rng=None):
  del rng
  width, height = camera_size
  nv12 = NV12Frame(width, height, *get_nv12_info(width, height))
  frame_size = nv12_copy_size(nv12.stride, nv12.y_height, nv12.uv_height)
  return {
    'frame': Tensor.randint((frame_size,), low=0, high=256, dtype=dtypes.uint8, device=device).realize(),
    'big_frame': Tensor.randint((frame_size,), low=0, high=256, dtype=dtypes.uint8, device=device).realize(),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description='Compile a standalone 3X camera warp JIT')
  parser.add_argument('--camera-size', type=parse_size, default=(1928, 1208))
  parser.add_argument('--model-size', type=parse_size, default=(512, 256))
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--benchmark-runs', type=int, default=3)
  args = parser.parse_args()
  if args.benchmark_runs < 1:
    parser.error('--benchmark-runs must be at least 1')

  cam_w, cam_h = args.camera_size
  model_w, model_h = args.model_size
  nv12 = NV12Frame(cam_w, cam_h, *get_nv12_info(cam_w, cam_h))
  warp = make_warp(nv12, model_w, model_h)
  queues = partial(make_queues, args.camera_size)
  random_frames = partial(make_random_frames, args.camera_size, Device.DEFAULT)
  jit = compile_jit(TinyJit(warp, prune=True), WARP_INPUTS, queues,
                    make_random_inputs=random_frames, benchmark_runs=args.benchmark_runs)
  artifact = {
    'camera_size': args.camera_size,
    'model_size': args.model_size,
    'input_device': Device.DEFAULT,
    'warp': jit,
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open('wb') as output_file:
    dump_oob(artifact, output_file)
  print(f'Warp artifact: {args.output} ({args.output.stat().st_size / 1e6:.2f} MB)')


if __name__ == '__main__':
  main()
