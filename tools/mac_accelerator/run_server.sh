#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PORT="${PORT:-8066}"
ARTIFACT="${ARTIFACT:-$SCRIPT_DIR/artifacts/driving_policy_metal.pkl}"

LINK_INFO="$("$SCRIPT_DIR/setup_usb_ncm.sh")"
printf '%s\n' "$LINK_INFO"
MAC_ENDPOINT="$(printf '%s\n' "$LINK_INFO" | awk '/^Mac:/{print $2; exit}')"
if [[ -z "$MAC_ENDPOINT" ]]; then
  echo "Could not discover the Mac USB endpoint." >&2
  exit 1
fi

cd "$REPO_ROOT"
DEV=METAL JIT=2 exec "$REPO_ROOT/.venv/bin/python" \
  "$SCRIPT_DIR/inference_server.py" --host "$MAC_ENDPOINT" --port "$PORT" --artifact "$ARTIFACT"
