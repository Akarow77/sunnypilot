#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-$SCRIPT_DIR/artifacts}"
OUTPUT="${OUTPUT:-$ARTIFACT_DIR/driving_tinygrad_metal.pkl}"
BENCHMARK_RUNS="${BENCHMARK_RUNS:-10}"

if [[ ! -x "$REPO_ROOT/.venv/bin/python" ]]; then
  echo "Missing $REPO_ROOT/.venv. Run: uv sync --frozen --all-extras" >&2
  exit 1
fi

MODEL="${MODEL:-$REPO_ROOT/openpilot/selfdrive/modeld/models/driving_supercombo.onnx}"
if [[ "$(wc -c < "$MODEL")" -lt 1000000 ]]; then
  echo "The driving model is still a Git LFS pointer." >&2
  echo "Run: source .venv/bin/activate && git lfs pull --include=openpilot/selfdrive/modeld/models/driving_supercombo.onnx" >&2
  exit 1
fi

mkdir -p "$ARTIFACT_DIR"
cd "$REPO_ROOT"

# JIT=2 is required on M1/M2. Metal's indirect-command-buffer graph replay
# currently fails sunnypilot's pickle round-trip correctness check on M2.
DEV=METAL JIT=2 "$REPO_ROOT/.venv/bin/python" \
  "$REPO_ROOT/openpilot/selfdrive/modeld/compile_modeld.py" \
  --model-size 512x256 \
  --camera-resolutions 1928x1208 \
  --onnx "$MODEL" \
  --output "$OUTPUT" \
  --frame-skip 4 \
  --benchmark-runs "$BENCHMARK_RUNS"

echo "Metal artifact: $OUTPUT"
