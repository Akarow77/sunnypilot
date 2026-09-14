#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="$REPO_ROOT/.coreml-venv/bin/python"

cd "$REPO_ROOT"
if [[ ! -x "$PYTHON" ]]; then
  uv venv --python 3.12 .coreml-venv
fi
uv pip install --python "$PYTHON" \
  'coremltools==9.0' 'lz4==4.4.5' 'onnx==1.22.0' 'onnx2torch==1.5.15' \
  'torch==2.7.0' 'torchvision==0.22.0' 'zstandard==0.25.0'
