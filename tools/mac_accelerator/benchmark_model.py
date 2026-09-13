#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import time

import numpy as np

from tinygrad.device import Device

from openpilot.selfdrive.modeld.compile_modeld import MODELD_INPUTS, make_input_queues, nv12_copy_size
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.helpers import load_oob
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info


DEFAULT_ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "driving_tinygrad_metal.pkl"
CAMERA_SIZE = (1928, 1208)


def percentile(values: list[float], q: float) -> float:
  return float(np.percentile(np.asarray(values), q))


def run_once(run_model, input_queues, npy, hidden_state_slice: slice) -> tuple[float, np.ndarray]:
  started = time.perf_counter()
  outputs, = run_model(**{name: input_queues[name] for name in MODELD_INPUTS})
  model_output = outputs.numpy()[0]
  npy['prev_feat'][:] = model_output[hidden_state_slice]
  elapsed_ms = (time.perf_counter() - started) * 1e3
  if not np.all(np.isfinite(model_output)):
    raise RuntimeError("model produced non-finite output")
  return elapsed_ms, model_output


def benchmark(artifact_path: Path, duration: float, warmup_runs: int, deadline_ms: float) -> dict:
  with artifact_path.open("rb") as artifact_file:
    artifact = load_oob(artifact_file)

  cam_w, cam_h = CAMERA_SIZE
  frame_copy_size = nv12_copy_size(*get_nv12_info(cam_w, cam_h)[:3])
  frame_skip = ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ
  metadata = artifact['metadata']
  hidden_state_slice = metadata['output_slices']['hidden_state']
  input_queues, npy, frame_views = make_input_queues(
    metadata['input_shapes'], frame_skip, artifact['input_devices']['model'], frame_copy_size)
  run_model = artifact['run_model'][CAMERA_SIZE]

  rng = np.random.default_rng(42)
  for frame in frame_views.values():
    frame[:] = rng.integers(0, 256, size=frame.shape, dtype=np.uint8)
  for value in npy.values():
    value[:] = 0
  npy['tfm'][:] = np.eye(3, dtype=np.float32)
  npy['big_tfm'][:] = np.eye(3, dtype=np.float32)

  for _ in range(warmup_runs):
    run_once(run_model, input_queues, npy, hidden_state_slice)

  timings: list[float] = []
  started = time.monotonic()
  while time.monotonic() - started < duration:
    elapsed_ms, _ = run_once(run_model, input_queues, npy, hidden_state_slice)
    timings.append(elapsed_ms)

  midpoint = max(1, len(timings) // 2)
  misses = sum(t > deadline_ms for t in timings)
  first_half_mean = float(np.mean(timings[:midpoint]))
  second_half_mean = float(np.mean(timings[midpoint:]))
  result = {
    "device": Device.DEFAULT,
    "camera": f"{cam_w}x{cam_h}",
    "duration_s": round(time.monotonic() - started, 3),
    "runs": len(timings),
    "deadline_ms": deadline_ms,
    "deadline_misses": misses,
    "deadline_miss_percent": 100.0 * misses / len(timings),
    "mean_ms": float(np.mean(timings)),
    "p50_ms": percentile(timings, 50),
    "p95_ms": percentile(timings, 95),
    "p99_ms": percentile(timings, 99),
    "max_ms": max(timings),
    "first_half_mean_ms": first_half_mean,
    "second_half_mean_ms": second_half_mean,
    "second_vs_first_half_percent": 100.0 * (second_half_mean / first_half_mean - 1.0),
  }
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description="Sustained sunnypilot model benchmark for a comma 3X camera on Apple Metal")
  parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
  parser.add_argument("--duration", type=float, default=60.0)
  parser.add_argument("--warmup-runs", type=int, default=10)
  parser.add_argument("--deadline-ms", type=float, default=50.0)
  parser.add_argument("--json-output", type=Path)
  args = parser.parse_args()

  if args.duration <= 0:
    parser.error("--duration must be positive")
  if args.warmup_runs < 1:
    parser.error("--warmup-runs must be at least 1")
  if not args.artifact.is_file():
    parser.error(f"artifact not found: {args.artifact}; run compile_model.sh first")

  result = benchmark(args.artifact, args.duration, args.warmup_runs, args.deadline_ms)
  rendered = json.dumps(result, indent=2)
  print(rendered)
  if args.json_output is not None:
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(rendered + "\n")


if __name__ == "__main__":
  main()
