#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMES="${FRAMES:-200}"
WARMUP_FRAMES="${WARMUP_FRAMES:-5}"
FREQUENCY="${FREQUENCY:-20}"
DEADLINE_MS="${DEADLINE_MS:-50}"
PORT="${PORT:-8066}"
DEVICE_DIR="/tmp/mac_accelerator"

LINK_INFO="$("$SCRIPT_DIR/setup_usb_ncm.sh")"
printf '%s\n' "$LINK_INFO"
MAC_ENDPOINT="$(printf '%s\n' "$LINK_INFO" | awk '/^Mac:/{print $2; exit}')"
MAC_ADDRESS="${MAC_ENDPOINT%\%*}"
DEVICE_ENDPOINT="${MAC_ADDRESS}%usb0"

adb shell "mkdir -p '$DEVICE_DIR'"
adb push "$SCRIPT_DIR/transport.py" "$SCRIPT_DIR/synthetic_client.py" "$DEVICE_DIR/"
adb shell "cd '$DEVICE_DIR' && python3 synthetic_client.py '$DEVICE_ENDPOINT' \
  --port '$PORT' --frames '$FRAMES' --warmup-frames '$WARMUP_FRAMES' \
  --frequency '$FREQUENCY' --deadline-ms '$DEADLINE_MS'"
