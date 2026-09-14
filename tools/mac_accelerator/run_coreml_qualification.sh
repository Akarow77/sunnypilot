#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL="${MODEL:-$SCRIPT_DIR/artifacts/big_driving.mlpackage}"
METADATA="${METADATA:-$REPO_ROOT/openpilot/selfdrive/modeld/models/big_driving_supercombo_metadata.pkl}"
DURATION="${DURATION:-300}"
FREQUENCY="${FREQUENCY:-20}"
DEADLINE_MS="${DEADLINE_MS:-50}"
REPORT="${REPORT:-$SCRIPT_DIR/artifacts/coreml-qualification.json}"
COREML_TMPDIR="${COREML_TMPDIR:-$SCRIPT_DIR/artifacts/tmp}"

mkdir -p "$COREML_TMPDIR"
export TMPDIR="$COREML_TMPDIR"
export PYTHONUNBUFFERED=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

exec caffeinate -dimsu "$REPO_ROOT/.coreml-venv/bin/python" \
  "$SCRIPT_DIR/benchmark_coreml.py" "$MODEL" --metadata "$METADATA" \
  --compute-unit cpu-and-ne --warmup 40 --frequency "$FREQUENCY" \
  --duration "$DURATION" --deadline-ms "$DEADLINE_MS" \
  --json-output "$REPORT" --require-zero-misses
