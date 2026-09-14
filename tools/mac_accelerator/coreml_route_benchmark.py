#!/usr/bin/env python3
"""Replay comma camera segments through the Mac Metal warp and Core ML Big Model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import pickle
import subprocess
import time

import coremltools as ct
import numpy as np

from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.tools.lib.logreader import LogReader

from coreml_inference_server import CoreMLPolicySession
from macos_performance import configure_user_interactive_qos
from route_model_benchmark import CAMERA_SIZE, Nv12Decoder, copy_padded_nv12, percentile, route_context
from transport import POLICY_INPUTS, WARPED_BYTES
from warp_runtime import WarpRuntime


FRAME_DELAY = 0.05
ACTION_DELAY = 0.025
LONG_SMOOTH_SECONDS = 0.3


def video_frame_count(path: Path) -> int:
  command = [
    'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
    '-show_entries', 'stream=nb_read_frames', '-of', 'default=nokey=1:noprint_wrappers=1', str(path),
  ]
  return int(subprocess.check_output(command, text=True).strip())


def route_key(route_dir: Path) -> str:
  return route_dir.name.rsplit('--', 1)[0]


def summarize(values: list[float]) -> dict[str, float]:
  return {
    'mean_ms': float(np.mean(values)),
    'p50_ms': percentile(values, 50),
    'p95_ms': percentile(values, 95),
    'p99_ms': percentile(values, 99),
    'max_ms': max(values),
  }


def route_reference(log_path: Path) -> tuple[list[int], dict[int, tuple[float, float, float, bool]], float, float]:
  encoded_frame_ids = []
  model_actions = {}
  v_ego = 0.0
  lateral_delay = None
  longitudinal_delay = None
  for message in LogReader(str(log_path)):
    which = message.which()
    if which == 'narrowRoadEncodeIdx':
      encoded_frame_ids.append(int(message.narrowRoadEncodeIdx.frameId))
    elif which == 'carState':
      v_ego = float(message.carState.vEgo)
    elif which == 'lateralDelay':
      lateral_delay = float(message.lateralDelay.lateralDelay)
    elif which == 'carParams':
      longitudinal_delay = float(message.carParams.longitudinalActuatorDelay)
    elif which == 'modelV2':
      model_state = message.modelV2
      model_actions[int(model_state.frameId)] = (
        v_ego,
        float(model_state.action.desiredCurvature),
        float(model_state.action.desiredAcceleration),
        bool(model_state.big),
      )
  if lateral_delay is None or longitudinal_delay is None:
    raise RuntimeError('route is missing lateralDelay or carParams')
  return encoded_frame_ids, model_actions, lateral_delay, longitudinal_delay


def comparison_summary(big_curvatures: list[float], local_curvatures: list[float],
                       big_lateral_actions: list[float], local_lateral_actions: list[float]) -> dict:
  big_curvature = np.asarray(big_curvatures)
  local_curvature = np.asarray(local_curvatures)
  big_lateral = np.asarray(big_lateral_actions)
  local_lateral = np.asarray(local_lateral_actions)
  curvature_error = np.abs(big_curvature - local_curvature)
  lateral_error = np.abs(big_lateral - local_lateral)

  def correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
      return None
    return float(np.corrcoef(left, right)[0, 1])

  return {
    'frames': len(big_curvature),
    'minimum_speed_mps': 5.0,
    'curvature_mae': float(np.mean(curvature_error)),
    'curvature_p95_abs_error': float(np.percentile(curvature_error, 95)),
    'curvature_max_abs_error': float(np.max(curvature_error)),
    'curvature_correlation': correlation(big_curvature, local_curvature),
    'lateral_action_mae': float(np.mean(lateral_error)),
    'lateral_action_p95_abs_error': float(np.percentile(lateral_error, 95)),
    'lateral_action_max_abs_error': float(np.max(lateral_error)),
    'lateral_action_correlation': correlation(big_lateral, local_lateral),
  }


def benchmark(route_dirs: list[Path], model_path: Path, metadata_path: Path,
              warp_path: Path, deadline_ms: float, frame_limit: int) -> dict:
  for path in [model_path, metadata_path, warp_path, *route_dirs]:
    if not path.exists():
      raise FileNotFoundError(path)
  with metadata_path.open('rb') as metadata_file:
    metadata = pickle.load(metadata_file)

  qos_enabled = configure_user_interactive_qos()
  model = ct.models.MLModel(str(model_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
  output_name = model.get_spec().description.output[0].name
  output_floats = math.prod(metadata['output_shapes']['outputs'])
  action_slice = metadata['output_slices'].get('action')
  warp = WarpRuntime(warp_path)
  width, height = CAMERA_SIZE
  if warp.camera_size != CAMERA_SIZE:
    raise RuntimeError(f'warp camera size {warp.camera_size} != {CAMERA_SIZE}')
  padded_narrow = np.zeros(warp.frame_size, dtype=np.uint8)
  padded_wide = np.zeros(warp.frame_size, dtype=np.uint8)
  stride, y_height, uv_height, _ = get_nv12_info(width, height)

  # Compile the Metal warp before measuring.
  identity = np.eye(3, dtype=np.float32)
  for _ in range(3):
    warp.run(padded_narrow, padded_wide, identity, identity)

  route_reports = []
  all_warp_ms: list[float] = []
  all_inference_ms: list[float] = []
  all_compute_ms: list[float] = []
  all_big_curvatures: list[float] = []
  all_local_curvatures: list[float] = []
  all_big_lateral_actions: list[float] = []
  all_local_lateral_actions: list[float] = []
  previous_key = None
  session = None
  comparison_started = False
  total_processed = 0

  for route_dir in route_dirs:
    fcamera = route_dir / 'fcamera.hevc'
    ecamera = route_dir / 'ecamera.hevc'
    rlog = route_dir / 'rlog.zst'
    for path in (fcamera, ecamera, rlog):
      if not path.is_file():
        raise FileNotFoundError(path)
    calibration, is_rhd, device_type, sensor = route_context(rlog)
    encoded_frame_ids, model_actions, lateral_delay, longitudinal_delay = route_reference(rlog)
    camera = DEVICE_CAMERAS[(device_type, sensor)]
    narrow_transform = get_warp_matrix(calibration, camera.narrow_road.intrinsics, False).astype(np.float32)
    wide_transform = get_warp_matrix(calibration, camera.wide_road.intrinsics, True).astype(np.float32)
    key = route_key(route_dir)
    if key != previous_key:
      session = CoreMLPolicySession(model, metadata, output_name, frame_skip=2)
      zero_payload = bytes(WARPED_BYTES) + POLICY_INPUTS.pack(*([0.0] * 12))
      for _ in range(3):
        session.infer(zero_payload)
      session.reset()
      comparison_started = False
      previous_key = key
    assert session is not None
    traffic = (0.0, 1.0) if is_rhd else (1.0, 0.0)
    lat_action_t = lateral_delay + FRAME_DELAY + ACTION_DELAY
    long_action_t = longitudinal_delay + LONG_SMOOTH_SECONDS + FRAME_DELAY + ACTION_DELAY
    policy = POLICY_INPUTS.pack(*([0.0] * 8), *traffic, lat_action_t, long_action_t)
    available = min(video_frame_count(fcamera), video_frame_count(ecamera))
    count = available if frame_limit == 0 else min(available, frame_limit)
    route_warp_ms: list[float] = []
    route_inference_ms: list[float] = []
    route_compute_ms: list[float] = []
    route_big_curvatures: list[float] = []
    route_local_curvatures: list[float] = []
    route_big_lateral_actions: list[float] = []
    route_local_lateral_actions: list[float] = []
    action_abs_max = 0.0

    with Nv12Decoder(fcamera, width, height) as narrow_decoder, Nv12Decoder(ecamera, width, height) as wide_decoder:
      for frame_index in range(count):
        copy_padded_nv12(narrow_decoder.read(), padded_narrow, width, height, stride, y_height, uv_height)
        copy_padded_nv12(wide_decoder.read(), padded_wide, width, height, stride, y_height, uv_height)
        frame_id = encoded_frame_ids[frame_index]
        if not comparison_started and frame_id in model_actions:
          session.reset()
          comparison_started = True
        started_ns = time.monotonic_ns()
        warped, warp_ms = warp.run(padded_narrow, padded_wide, narrow_transform, wide_transform)
        inference_started_ns = time.monotonic_ns()
        output = session.infer(warped + policy)
        completed_ns = time.monotonic_ns()
        inference_ms = (completed_ns - inference_started_ns) / 1e6
        compute_ms = (completed_ns - started_ns) / 1e6
        values = np.frombuffer(output, dtype='<f2')
        if len(values) != output_floats or not np.all(np.isfinite(values)):
          raise RuntimeError(f'invalid model output at {route_dir.name} frame {frame_index}')
        if action_slice is not None:
          action_abs_max = max(action_abs_max, float(np.max(np.abs(values[action_slice]))))
          reference = model_actions.get(frame_id)
          if reference is not None:
            v_ego, local_curvature, _, _ = reference
            if v_ego >= 5.0:
              speed_scale = max(1.0, v_ego) ** 2
              big_lateral_action = float(values[action_slice.start])
              route_big_curvatures.append(big_lateral_action / speed_scale)
              route_local_curvatures.append(local_curvature)
              route_big_lateral_actions.append(big_lateral_action)
              route_local_lateral_actions.append(local_curvature * speed_scale)
        route_warp_ms.append(warp_ms)
        route_inference_ms.append(inference_ms)
        route_compute_ms.append(compute_ms)
        total_processed += 1
        if total_processed % 100 == 0:
          print(f'processed={total_processed} segment={route_dir.name} frame={frame_index + 1}/{count}', flush=True)

    misses = sum(value > deadline_ms for value in route_compute_ms)
    report = {
      'route_dir': str(route_dir),
      'frames': count,
      'available_frames': available,
      'device_type': device_type,
      'sensor': sensor,
      'lat_action_t': lat_action_t,
      'long_action_t': long_action_t,
      'action_abs_max': action_abs_max,
      'deadline_misses': misses,
      'deadline_miss_percent': 100.0 * misses / count,
      'warp': summarize(route_warp_ms),
      'inference': summarize(route_inference_ms),
      'compute': summarize(route_compute_ms),
      'steering_comparison': comparison_summary(
        route_big_curvatures, route_local_curvatures,
        route_big_lateral_actions, route_local_lateral_actions,
      ) if route_big_curvatures else None,
    }
    route_reports.append(report)
    all_warp_ms.extend(route_warp_ms)
    all_inference_ms.extend(route_inference_ms)
    all_compute_ms.extend(route_compute_ms)
    all_big_curvatures.extend(route_big_curvatures)
    all_local_curvatures.extend(route_local_curvatures)
    all_big_lateral_actions.extend(route_big_lateral_actions)
    all_local_lateral_actions.extend(route_local_lateral_actions)

  total_misses = sum(value > deadline_ms for value in all_compute_ms)
  return {
    'model': str(model_path),
    'warp': str(warp_path),
    'backend': 'COREML_ANE',
    'qos': 'user-interactive' if qos_enabled else 'default',
    'output_floats': output_floats,
    'deadline_ms': deadline_ms,
    'frames': len(all_compute_ms),
    'deadline_misses': total_misses,
    'deadline_miss_percent': 100.0 * total_misses / len(all_compute_ms),
    'warp_timing': summarize(all_warp_ms),
    'inference_timing': summarize(all_inference_ms),
    'compute_timing': summarize(all_compute_ms),
    'steering_comparison': comparison_summary(
      all_big_curvatures, all_local_curvatures,
      all_big_lateral_actions, all_local_lateral_actions,
    ) if all_big_curvatures else None,
    'routes': route_reports,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description='Replay comma route cameras through Core ML Big Model')
  parser.add_argument('route_dirs', type=Path, nargs='+')
  parser.add_argument('--model', type=Path, required=True)
  parser.add_argument('--metadata', type=Path, required=True)
  parser.add_argument('--warp', type=Path, required=True)
  parser.add_argument('--deadline-ms', type=float, default=55.0)
  parser.add_argument('--frames-per-segment', type=int, default=0,
                      help='zero replays every available frame')
  parser.add_argument('--json-output', type=Path)
  args = parser.parse_args()
  if args.deadline_ms <= 0 or args.frames_per_segment < 0:
    parser.error('deadline must be positive and frame limit cannot be negative')
  report = benchmark(args.route_dirs, args.model, args.metadata, args.warp,
                     args.deadline_ms, args.frames_per_segment)
  rendered = json.dumps(report, indent=2)
  print(rendered)
  if args.json_output is not None:
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(rendered + '\n')


if __name__ == '__main__':
  main()
