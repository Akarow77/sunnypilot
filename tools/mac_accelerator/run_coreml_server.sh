#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL="${MODEL:-$SCRIPT_DIR/artifacts/big_driving.mlpackage}"
ONNX="${ONNX:-$REPO_ROOT/openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx}"
METADATA="${METADATA:-$REPO_ROOT/openpilot/selfdrive/modeld/models/big_driving_supercombo_metadata.pkl}"
PORT="${PORT:-8066}"
COREML_TMPDIR="${COREML_TMPDIR:-$SCRIPT_DIR/artifacts/tmp}"

if [[ -z "${AUTH_KEY_FILE:-}" ]] || [[ ! -f "$AUTH_KEY_FILE" ]]; then
  echo "AUTH_KEY_FILE must point to a shared key containing at least 32 bytes." >&2
  exit 1
fi

mkdir -p "$COREML_TMPDIR"
export TMPDIR="$COREML_TMPDIR"

if [[ "${MAC_ACCELERATOR_LOCAL_TEST:-0}" == "1" ]]; then
  MAC_ENDPOINT="::1"
  echo "Mac-only test mode: localhost; no comma connection required."
fi

cd "$REPO_ROOT"
SERVER=("$REPO_ROOT/.coreml-venv/bin/python" "$SCRIPT_DIR/coreml_inference_server.py"
  --model "$MODEL" --metadata "$METADATA" --onnx "$ONNX"
  --port "$PORT" --auth-key-file "$AUTH_KEY_FILE")
if [[ "${MAC_ACCELERATOR_LOCAL_TEST:-0}" == "1" ]]; then
  exec "${SERVER[@]}" --host "$MAC_ENDPOINT"
fi
exec "$REPO_ROOT/.coreml-venv/bin/python" "$SCRIPT_DIR/usb_ncm_supervisor.py" \
  --setup "$SCRIPT_DIR/setup_usb_ncm.sh" -- "${SERVER[@]}"
