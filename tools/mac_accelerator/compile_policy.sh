#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-$SCRIPT_DIR/artifacts}"
OUTPUT="${OUTPUT:-$ARTIFACT_DIR/driving_policy_metal.pkl}"
BENCHMARK_RUNS="${BENCHMARK_RUNS:-10}"
MODEL="$REPO_ROOT/openpilot/selfdrive/modeld/models/driving_supercombo.onnx"

if [[ ! -x "$REPO_ROOT/.venv/bin/python" ]]; then
  echo "Missing $REPO_ROOT/.venv. Run: uv sync --frozen --all-extras" >&2
  exit 1
fi
if [[ "$(wc -c < "$MODEL")" -lt 1000000 ]]; then
  echo "The driving model is still a Git LFS pointer." >&2
  exit 1
fi

cd "$REPO_ROOT"
DEV=METAL JIT=2 exec "$REPO_ROOT/.venv/bin/python" \
  "$SCRIPT_DIR/compile_policy.py" \
  --onnx "$MODEL" \
  --output "$OUTPUT" \
  --benchmark-runs "$BENCHMARK_RUNS"
