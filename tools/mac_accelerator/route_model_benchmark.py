#!/usr/bin/env python3
"""Replay local comma camera logs through a compiled full-model artifact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np

from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.selfdrive.modeld.compile_modeld import MODELD_INPUTS, make_input_queues, nv12_copy_size
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.helpers import load_oob
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.tools.lib.logreader import LogReader


CAMERA_SIZE = (1928, 1208)


class Nv12Decoder:
  def __init__(self, path: Path, width: int, height: int):
    self.frame_bytes = width * height * 3 // 2
    command = [
      'ffmpeg', '-v', 'error', '-nostdin', '-threads', '0', '-hwaccel', 'none',
      '-c:v', 'hevc', '-f', 'hevc', '-i', str(path), '-fps_mode', 'passthrough',
      '-f', 'rawvideo', '-pix_fmt', 'nv12', 'pipe:1',
    ]
    self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

  def read(self) -> bytes:
    if self.process.stdout is None:
      raise RuntimeError('ffmpeg stdout is unavailable')
    data = bytearray(self.frame_bytes)
    view = memoryview(data)
    received = 0
    while received < self.frame_bytes:
      count = self.process.stdout.readinto(view[received:])
      if not count:
        raise EOFError(f'video ended after {received} of {self.frame_bytes} frame bytes')
      received += count
    return bytes(data)

  def close(self) -> None:
    if self.process.stdout is not None:
      self.process.stdout.close()
    if self.process.poll() is None:
      self.process.terminate()
    self.process.wait(timeout=5)

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback) -> None:
    self.close()


def route_context(log_path: Path) -> tuple[np.ndarray, bool, str, str]:
  device_type = None
  sensor = None
  calibration = None
  is_rhd = False
  for message in LogReader(str(log_path)):
    which = message.which()
    if which == 'deviceState' and device_type is None:
      device_type = str(message.deviceState.deviceType)
    elif which == 'narrowRoadCameraState' and sensor is None:
      sensor = str(message.narrowRoadCameraState.sensor)
    elif which == 'extrinsicsCalibration' and len(message.extrinsicsCalibration.rpyCalib) == 3:
      calibration = np.asarray(message.extrinsicsCalibration.rpyCalib, dtype=np.float32)
    elif which == 'driverMonitoringState':
      is_rhd = bool(message.driverMonitoringState.isRHD)
    if device_type is not None and sensor is not None and calibration is not None:
      break
  if device_type is None or sensor is None or calibration is None:
    raise RuntimeError('route is missing device, camera, or calibration messages')
  return calibration, is_rhd, device_type, sensor


def copy_padded_nv12(raw: bytes, destination: np.ndarray, width: int, height: int,
                     stride: int, y_height: int, uv_height: int) -> None:
  source = np.frombuffer(raw, dtype=np.uint8)
  y_bytes = width * height
  destination[:] = 0
  destination[:stride * y_height].reshape(y_height, stride)[:height, :width] = source[:y_bytes].reshape(height, width)
  destination[stride * y_height:].reshape(uv_height, stride)[:height // 2, :width] = source[y_bytes:].reshape(height // 2, width)


def percentile(values: list[float], q: float) -> float:
  return float(np.percentile(np.asarray(values), q))


def benchmark(route_dir: Path, artifact_path: Path, frames: int, warmup_frames: int, deadline_ms: float) -> dict:
  fcamera = route_dir / 'fcamera.hevc'
  ecamera = route_dir / 'ecamera.hevc'
  rlog = route_dir / 'rlog.zst'
  for path in (fcamera, ecamera, rlog, artifact_path):
    if not path.is_file():
      raise FileNotFoundError(path)

  calibration, is_rhd, device_type, sensor = route_context(rlog)
  camera = DEVICE_CAMERAS[(device_type, sensor)]
  transforms = {
    'img': get_warp_matrix(calibration, camera.narrow_road.intrinsics, False).astype(np.float32),
    'big_img': get_warp_matrix(calibration, camera.wide_road.intrinsics, True).astype(np.float32),
  }

  with artifact_path.open('rb') as artifact_file:
    artifact = load_oob(artifact_file)
  width, height = CAMERA_SIZE
  stride, y_height, uv_height, _ = get_nv12_info(width, height)
  frame_copy_size = nv12_copy_size(stride, y_height, uv_height)
  frame_skip = ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ
  queues, npy, frame_views = make_input_queues(
    artifact['metadata']['input_shapes'], frame_skip, artifact['input_devices']['model'], frame_copy_size)
  runner = artifact['run_model'][CAMERA_SIZE]
  output_slices = artifact['metadata']['output_slices']
  npy['desire'][:] = 0
  npy['traffic_convention'][:] = (0.0, 1.0) if is_rhd else (1.0, 0.0)
  npy['action_t'][:] = (0.1, 0.3)
  npy['tfm'][:] = transforms['img']
  npy['big_tfm'][:] = transforms['big_img']

  timings: list[float] = []
  action_abs_max = 0.0
  total_frames = warmup_frames + frames
  with Nv12Decoder(fcamera, width, height) as narrow_decoder, Nv12Decoder(ecamera, width, height) as wide_decoder:
    for frame_index in range(total_frames):
      copy_padded_nv12(narrow_decoder.read(), frame_views['img'], width, height, stride, y_height, uv_height)
      copy_padded_nv12(wide_decoder.read(), frame_views['big_img'], width, height, stride, y_height, uv_height)
      started = time.perf_counter()
      output, = runner(**{name: queues[name] for name in MODELD_INPUTS})
      result = output.numpy()[0]
      elapsed_ms = (time.perf_counter() - started) * 1e3
      if not np.all(np.isfinite(result)):
        raise RuntimeError(f'non-finite output at frame {frame_index}')
      npy['prev_feat'][:] = result[output_slices['hidden_state']]
      if 'action' in output_slices:
        action_abs_max = max(action_abs_max, float(np.max(np.abs(result[output_slices['action']]))))
      if frame_index >= warmup_frames:
        timings.append(elapsed_ms)

  misses = sum(value > deadline_ms for value in timings)
  return {
    'route_dir': str(route_dir),
    'artifact': str(artifact_path),
    'device': artifact['input_devices']['model'],
    'camera': f'{width}x{height}',
    'device_type': device_type,
    'sensor': sensor,
    'frames': frames,
    'warmup_frames': warmup_frames,
    'output_floats': math.prod(next(iter(artifact['metadata']['output_shapes'].values()))),
    'deadline_ms': deadline_ms,
    'deadline_misses': misses,
    'deadline_miss_percent': 100.0 * misses / frames,
    'mean_ms': float(np.mean(timings)),
    'p50_ms': percentile(timings, 50),
    'p95_ms': percentile(timings, 95),
    'p99_ms': percentile(timings, 99),
    'max_ms': max(timings),
    'action_abs_max': action_abs_max,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description='Replay comma route cameras through a Metal model artifact')
  parser.add_argument('route_dir', type=Path)
  parser.add_argument('--artifact', type=Path, required=True)
  parser.add_argument('--frames', type=int, default=200)
  parser.add_argument('--warmup-frames', type=int, default=5)
  parser.add_argument('--deadline-ms', type=float, default=50.0)
  parser.add_argument('--json-output', type=Path)
  args = parser.parse_args()
  if args.frames < 1 or args.warmup_frames < 0 or args.deadline_ms <= 0:
    parser.error('frames and deadline must be positive; warmup frames cannot be negative')
  result = benchmark(args.route_dir, args.artifact, args.frames, args.warmup_frames, args.deadline_ms)
  rendered = json.dumps(result, indent=2)
  print(rendered)
  if args.json_output is not None:
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(rendered + '\n')


if __name__ == '__main__':
  main()
