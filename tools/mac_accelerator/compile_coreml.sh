#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ONNX="${ONNX:-$REPO_ROOT/openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx}"
METADATA="${METADATA:-$REPO_ROOT/openpilot/selfdrive/modeld/models/big_driving_supercombo_metadata.pkl}"
OUTPUT="${OUTPUT:-$SCRIPT_DIR/artifacts/big_driving.mlpackage}"
COREML_PYTHON="$REPO_ROOT/.coreml-venv/bin/python"

if [[ ! -x "$COREML_PYTHON" ]]; then
  echo "Missing .coreml-venv; run $SCRIPT_DIR/setup_coreml_env.sh first." >&2
  exit 1
fi

cd "$REPO_ROOT"
DEV=CPU CACHEDB="$SCRIPT_DIR/artifacts/metadata-cache.db" \
  "$REPO_ROOT/.venv/bin/python" openpilot/sunnypilot/modeld_v2/get_model_metadata.py "$ONNX"
"$COREML_PYTHON" "$SCRIPT_DIR/convert_coreml.py" \
  --onnx "$ONNX" --promote-fp32 --output "$OUTPUT"
test -f "$METADATA"
