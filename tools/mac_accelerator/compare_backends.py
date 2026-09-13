#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from openpilot.selfdrive.modeld.compile_modeld import MODELD_INPUTS, make_input_queues, nv12_copy_size
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.helpers import load_oob
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info


CAMERA_SIZE = (1928, 1208)
DEFAULT_DIR = Path(__file__).resolve().parent / "artifacts"


def load_runner(path: Path):
  with path.open("rb") as artifact_file:
    artifact = load_oob(artifact_file)
  cam_w, cam_h = CAMERA_SIZE
  frame_copy_size = nv12_copy_size(*get_nv12_info(cam_w, cam_h)[:3])
  frame_skip = ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ
  queues, npy, frames = make_input_queues(
    artifact['metadata']['input_shapes'], frame_skip, artifact['input_devices']['model'], frame_copy_size)
  return artifact, queues, npy, frames


def fill_inputs(npy, frames, frame_values, transform, policy_values) -> None:
  for name, frame in frames.items():
    frame[:] = frame_values[name]
  npy['tfm'][:] = transform
  npy['big_tfm'][:] = transform
  npy['desire'][:] = policy_values['desire']
  npy['traffic_convention'][:] = policy_values['traffic_convention']
  npy['action_t'][:] = policy_values['action_t']


def infer(artifact, queues, npy) -> np.ndarray:
  runner = artifact['run_model'][CAMERA_SIZE]
  output, = runner(**{name: queues[name] for name in MODELD_INPUTS})
  result = output.numpy()[0]
  npy['prev_feat'][:] = result[artifact['metadata']['output_slices']['hidden_state']]
  return result


def compare(cpu_path: Path, metal_path: Path, frames_count: int) -> dict:
  cpu = load_runner(cpu_path)
  metal = load_runner(metal_path)
  cpu_artifact, cpu_queues, cpu_npy, cpu_frames = cpu
  metal_artifact, metal_queues, metal_npy, metal_frames = metal
  if cpu_artifact['metadata']['output_slices'] != metal_artifact['metadata']['output_slices']:
    raise RuntimeError("CPU and Metal artifacts have different output layouts")

  rng = np.random.default_rng(42)
  transform = np.eye(3, dtype=np.float32)
  for npy in (cpu_npy, metal_npy):
    for value in npy.values():
      value[:] = 0
  frame_metrics = []
  output_metrics: dict[str, list[dict[str, float | int]]] = {
    name: [] for name in cpu_artifact['metadata']['output_slices'] if name != 'pad'
  }
  for frame_index in range(frames_count):
    frame_values = {name: rng.integers(0, 256, size=value.shape, dtype=np.uint8) for name, value in cpu_frames.items()}
    policy_values = {
      'desire': np.zeros(cpu_npy['desire'].shape, dtype=np.float32),
      'traffic_convention': np.array([1.0, 0.0], dtype=np.float32),
      'action_t': np.array([0.1, 0.1], dtype=np.float32),
    }
    fill_inputs(cpu_npy, cpu_frames, frame_values, transform, policy_values)
    fill_inputs(metal_npy, metal_frames, frame_values, transform, policy_values)
    cpu_output = infer(cpu_artifact, cpu_queues, cpu_npy)
    metal_output = infer(metal_artifact, metal_queues, metal_npy)
    error = metal_output - cpu_output
    cpu_rms = float(np.sqrt(np.mean(np.square(cpu_output, dtype=np.float64))))
    frame_metrics.append({
      'frame': frame_index,
      'max_abs': float(np.max(np.abs(error))),
      'mean_abs': float(np.mean(np.abs(error))),
      'rmse': float(np.sqrt(np.mean(np.square(error, dtype=np.float64)))),
      'normalized_rmse': float(np.sqrt(np.mean(np.square(error, dtype=np.float64))) / max(cpu_rms, np.finfo(np.float32).tiny)),
    })
    for name in output_metrics:
      output_slice = cpu_artifact['metadata']['output_slices'][name]
      cpu_values = cpu_output[output_slice]
      output_error = error[output_slice]
      output_rmse = float(np.sqrt(np.mean(np.square(output_error, dtype=np.float64))))
      output_rms = float(np.sqrt(np.mean(np.square(cpu_values, dtype=np.float64))))
      output_metrics[name].append({
        'frame': frame_index,
        'max_abs': float(np.max(np.abs(output_error))),
        'normalized_rmse': output_rmse / max(output_rms, np.finfo(np.float32).tiny),
      })

  output_summary = {}
  for name, metrics in output_metrics.items():
    worst_max_abs = max(metrics, key=lambda item: item['max_abs'])
    worst_normalized_rmse = max(metrics, key=lambda item: item['normalized_rmse'])
    output_summary[name] = {
      'worst_max_abs': worst_max_abs['max_abs'],
      'worst_max_abs_frame': worst_max_abs['frame'],
      'worst_normalized_rmse': worst_normalized_rmse['normalized_rmse'],
      'worst_normalized_rmse_frame': worst_normalized_rmse['frame'],
    }

  return {
    'camera': f"{CAMERA_SIZE[0]}x{CAMERA_SIZE[1]}",
    'frames': frames_count,
    'frame_metrics': frame_metrics,
    'output_summary': output_summary,
    'worst_max_abs': max(item['max_abs'] for item in frame_metrics),
    'worst_normalized_rmse': max(item['normalized_rmse'] for item in frame_metrics),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description="Compare CPU and Metal sunnypilot JIT outputs over a recurrent frame sequence")
  parser.add_argument("--cpu-artifact", type=Path, default=DEFAULT_DIR / "driving_tinygrad_cpu.pkl")
  parser.add_argument("--metal-artifact", type=Path, default=DEFAULT_DIR / "driving_tinygrad_metal.pkl")
  parser.add_argument("--frames", type=int, default=10)
  parser.add_argument("--json-output", type=Path)
  args = parser.parse_args()
  if args.frames < 1:
    parser.error("--frames must be at least 1")
  for path in (args.cpu_artifact, args.metal_artifact):
    if not path.is_file():
      parser.error(f"artifact not found: {path}")

  result = compare(args.cpu_artifact, args.metal_artifact, args.frames)
  rendered = json.dumps(result, indent=2)
  print(rendered)
  if args.json_output is not None:
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(rendered + "\n")


if __name__ == "__main__":
  main()
