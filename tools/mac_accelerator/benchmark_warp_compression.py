#!/usr/bin/env python3
"""Measure lossless compression on warped tensors from a recorded route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import zlib

import numpy as np

from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.selfdrive.modeld.compile_modeld import nv12_copy_size
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

from route_model_benchmark import CAMERA_SIZE, Nv12Decoder, copy_padded_nv12, route_context
from transport import WARPED_BYTES
from warp_runtime import WarpRuntime


def main() -> None:
  parser = argparse.ArgumentParser(description='Benchmark zlib on real warped route tensors')
  parser.add_argument('route_dir', type=Path)
  parser.add_argument('--warp-artifact', type=Path, required=True)
  parser.add_argument('--frames', type=int, default=50)
  parser.add_argument('--json-output', type=Path)
  parser.add_argument('--sample-output', type=Path)
  parser.add_argument('--warps-output', type=Path, help='save all warped frames as a .npy array')
  args = parser.parse_args()
  if args.frames < 1:
    parser.error('--frames must be positive')

  calibration, _, device_type, sensor = route_context(args.route_dir / 'rlog.zst')
  camera = DEVICE_CAMERAS[(device_type, sensor)]
  narrow_transform = get_warp_matrix(calibration, camera.narrow_road.intrinsics, False).astype(np.float32)
  wide_transform = get_warp_matrix(calibration, camera.wide_road.intrinsics, True).astype(np.float32)
  width, height = CAMERA_SIZE
  stride, y_height, uv_height, _ = get_nv12_info(width, height)
  frame_size = nv12_copy_size(stride, y_height, uv_height)
  narrow = np.zeros(frame_size, dtype=np.uint8)
  wide = np.zeros(frame_size, dtype=np.uint8)
  runtime = WarpRuntime(args.warp_artifact)
  stats = {level: {'bytes': [], 'compress_ms': [], 'decompress_ms': []} for level in (1, 3, 6)}
  first_warped = None
  warped_frames = []

  with (Nv12Decoder(args.route_dir / 'fcamera.hevc', width, height) as narrow_decoder,
        Nv12Decoder(args.route_dir / 'ecamera.hevc', width, height) as wide_decoder):
    for _ in range(args.frames):
      copy_padded_nv12(narrow_decoder.read(), narrow, width, height, stride, y_height, uv_height)
      copy_padded_nv12(wide_decoder.read(), wide, width, height, stride, y_height, uv_height)
      warped, _ = runtime.run(narrow, wide, narrow_transform, wide_transform)
      if first_warped is None:
        first_warped = warped
      if args.warps_output is not None:
        warped_frames.append(np.frombuffer(warped, dtype=np.uint8).reshape(2, 6, 128, 256).copy())
      for level, values in stats.items():
        started = time.perf_counter()
        compressed = zlib.compress(warped, level)
        values['compress_ms'].append((time.perf_counter() - started) * 1000.0)
        started = time.perf_counter()
        restored = zlib.decompress(compressed)
        values['decompress_ms'].append((time.perf_counter() - started) * 1000.0)
        if restored != warped:
          raise RuntimeError('lossless compression round trip failed')
        values['bytes'].append(len(compressed))

  report = {
    str(level): {
      'compressed_bytes_mean': float(np.mean(values['bytes'])),
      'compression_ratio': float(np.mean(values['bytes'])) / WARPED_BYTES,
      'compress_ms_mean': float(np.mean(values['compress_ms'])),
      'decompress_ms_mean': float(np.mean(values['decompress_ms'])),
      'wire_mbps_at_20hz': float(np.mean(values['bytes'])) * 20 * 8 / 1e6,
    }
    for level, values in stats.items()
  }
  print(json.dumps(report, indent=2, sort_keys=True))
  if args.json_output is not None:
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
  if args.sample_output is not None:
    args.sample_output.parent.mkdir(parents=True, exist_ok=True)
    args.sample_output.write_bytes(first_warped)
  if args.warps_output is not None:
    args.warps_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.warps_output, np.stack(warped_frames))


if __name__ == '__main__':
  main()
