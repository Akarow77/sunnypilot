#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMES="${FRAMES:-200}"
WARMUP_FRAMES="${WARMUP_FRAMES:-20}"
QUALIFICATION_FRAMES="${QUALIFICATION_FRAMES:-20}"
FREQUENCY="${FREQUENCY:-20}"
DEADLINE_MS="${DEADLINE_MS:-50}"
QUALIFICATION_TIMEOUT_MS="${QUALIFICATION_TIMEOUT_MS:-500}"
PORT="${PORT:-8066}"
EXPECTED_OUTPUT_FLOATS="${EXPECTED_OUTPUT_FLOATS:-2576}"
EXPECTED_BACKEND="${EXPECTED_BACKEND:-METAL}"
EXPECTED_MODEL_SHA256="${EXPECTED_MODEL_SHA256:-}"
EXPECTED_CHECKPOINT="${EXPECTED_CHECKPOINT:-}"
COMPRESSION="${COMPRESSION:-}"
SYNTHETIC_RANDOM_PREFIX_BYTES="${SYNTHETIC_RANDOM_PREFIX_BYTES:-0}"
DEVICE_DIR="/tmp/mac_accelerator"
AUTH_ARGS=()
IDENTITY_ARGS=(--expected-backend "$EXPECTED_BACKEND")

if [[ "$(adb shell 'cat /data/params/d/IsOffroad 2>/dev/null' | tr -d '\r')" != "1" ]]; then
  echo "Refusing to run: comma 3X is not off-road." >&2
  exit 1
fi

LINK_INFO="$("$SCRIPT_DIR/setup_usb_ncm.sh")"
printf '%s\n' "$LINK_INFO"
MAC_ENDPOINT="$(printf '%s\n' "$LINK_INFO" | awk '/^Mac:/{print $2; exit}')"
MAC_ADDRESS="${MAC_ENDPOINT%\%*}"
DEVICE_ENDPOINT="${MAC_ADDRESS}%usb0"

adb shell "mkdir -p '$DEVICE_DIR'"
adb push "$SCRIPT_DIR/accelerator_client.py" "$SCRIPT_DIR/accelerator_protocol.py" \
  "$SCRIPT_DIR/compression.py" "$SCRIPT_DIR/fallback.py" "$SCRIPT_DIR/transport.py" \
  "$SCRIPT_DIR/synthetic_client.py" "$SCRIPT_DIR/ui_state.py" "$DEVICE_DIR/"
if [[ -n "${AUTH_KEY_FILE:-}" ]]; then
  adb push "$AUTH_KEY_FILE" "$DEVICE_DIR/auth.key"
  AUTH_ARGS=(--auth-key-file "$DEVICE_DIR/auth.key")
fi
if [[ -n "$EXPECTED_MODEL_SHA256" ]]; then
  IDENTITY_ARGS+=(--expected-model-sha256 "$EXPECTED_MODEL_SHA256")
fi
if [[ -n "$EXPECTED_CHECKPOINT" ]]; then
  IDENTITY_ARGS+=(--expected-checkpoint "$EXPECTED_CHECKPOINT")
fi
if [[ -n "$COMPRESSION" ]]; then
  IDENTITY_ARGS+=(--compression "$COMPRESSION")
fi
adb shell "cd '$DEVICE_DIR' && exec sudo -u comma -- env \
  PATH=/usr/comma/shims:/usr/local/venv/bin:/usr/local/bin:/usr/bin:/bin \
  PYTHONPATH=/data/openpilot python3 -u synthetic_client.py '$DEVICE_ENDPOINT' \
  --port '$PORT' --frames '$FRAMES' --warmup-frames '$WARMUP_FRAMES' \
  --qualification-frames '$QUALIFICATION_FRAMES' \
  --frequency '$FREQUENCY' --deadline-ms '$DEADLINE_MS' \
  --qualification-timeout-ms '$QUALIFICATION_TIMEOUT_MS' \
  --synthetic-random-prefix-bytes '$SYNTHETIC_RANDOM_PREFIX_BYTES' \
  --publish-ui-state \
  --realtime \
  --expected-output-floats '$EXPECTED_OUTPUT_FLOATS' ${IDENTITY_ARGS[*]} ${AUTH_ARGS[*]}"
