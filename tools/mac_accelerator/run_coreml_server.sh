#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL="${MODEL:-$SCRIPT_DIR/artifacts/big_driving.mlpackage}"
ONNX="${ONNX:-$REPO_ROOT/openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx}"
METADATA="${METADATA:-$REPO_ROOT/openpilot/selfdrive/modeld/models/big_driving_supercombo_metadata.pkl}"
PORT="${PORT:-8066}"

if [[ -z "${AUTH_KEY_FILE:-}" ]] || [[ ! -f "$AUTH_KEY_FILE" ]]; then
  echo "AUTH_KEY_FILE must point to a shared key containing at least 32 bytes." >&2
  exit 1
fi

LINK_INFO="$($SCRIPT_DIR/setup_usb_ncm.sh)"
printf '%s\n' "$LINK_INFO"
MAC_ENDPOINT="$(printf '%s\n' "$LINK_INFO" | awk '/^Mac:/{print $2; exit}')"
if [[ -z "$MAC_ENDPOINT" ]]; then
  echo "Could not discover the Mac USB endpoint." >&2
  exit 1
fi

cd "$REPO_ROOT"
exec "$REPO_ROOT/.coreml-venv/bin/python" "$SCRIPT_DIR/coreml_inference_server.py" \
  --model "$MODEL" --metadata "$METADATA" --onnx "$ONNX" \
  --host "$MAC_ENDPOINT" --port "$PORT" --auth-key-file "$AUTH_KEY_FILE"
