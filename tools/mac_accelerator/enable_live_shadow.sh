#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
AUTH_KEY_FILE="${AUTH_KEY_FILE:-$HOME/Library/Application Support/Sunnypilot Mac Accelerator/auth.key}"
MODEL="${MODEL:-$REPO_ROOT/openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx}"
DEADLINE_MS="${DEADLINE_MS:-55.0}"
PORT="${PORT:-8066}"
DEVICE_AUTH_DIR="/data/mac_accelerator"
DEVICE_AUTH_PATH="$DEVICE_AUTH_DIR/auth.key"

if [[ "$(adb shell 'cat /data/params/d/IsOffroad 2>/dev/null' | tr -d '\r')" != "1" ]]; then
  echo "Refusing to configure live shadow while the comma 3X is on-road." >&2
  exit 1
fi
if [[ ! -f "$AUTH_KEY_FILE" ]]; then
  echo "Authentication key not found: $AUTH_KEY_FILE" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "Big Model ONNX not found: $MODEL" >&2
  exit 1
fi

LINK_INFO="$($SCRIPT_DIR/setup_usb_ncm.sh)"
printf '%s\n' "$LINK_INFO"
MAC_ENDPOINT="$(printf '%s\n' "$LINK_INFO" | awk '/^Mac:/{print $2; exit}')"
MAC_ADDRESS="${MAC_ENDPOINT%\%*}"
DEVICE_ENDPOINT="${MAC_ADDRESS}%usb0"
MODEL_SHA256="$(shasum -a 256 "$MODEL" | awk '{print $1}')"

adb shell "mkdir -p '$DEVICE_AUTH_DIR' /data/media/0/mac_accelerator_shadow"
adb push "$AUTH_KEY_FILE" /data/local/tmp/mac-accelerator-auth.key >/dev/null
adb shell "install -o comma -g comma -m 600 /data/local/tmp/mac-accelerator-auth.key '$DEVICE_AUTH_PATH' && \
  rm /data/local/tmp/mac-accelerator-auth.key && \
  chown comma:comma /data/media/0/mac_accelerator_shadow"

adb shell "cd /data/openpilot && exec sudo -u comma -- env \
  PATH=/usr/comma/shims:/usr/local/venv/bin:/usr/local/bin:/usr/bin:/bin \
  PYTHONPATH=/data/openpilot \
  MAC_SHADOW_HOST='$DEVICE_ENDPOINT' MAC_SHADOW_PORT='$PORT' \
  MAC_SHADOW_DEADLINE='$DEADLINE_MS' MAC_SHADOW_SHA='$MODEL_SHA256' \
  python3 - <<'PY'
import os
from openpilot.common.params import Params

p = Params()
p.put('MacAcceleratorHost', os.environ['MAC_SHADOW_HOST'], block=True)
p.put('MacAcceleratorPort', int(os.environ['MAC_SHADOW_PORT']), block=True)
p.put('MacAcceleratorDeadlineMs', float(os.environ['MAC_SHADOW_DEADLINE']), block=True)
p.put('MacAcceleratorExpectedModelSHA256', os.environ['MAC_SHADOW_SHA'], block=True)
p.put_bool('MacAcceleratorShadowEnabled', True, block=True)
for key in ('MacAcceleratorPresent', 'MacAcceleratorLoading', 'MacAcceleratorReady',
            'MacAcceleratorActive', 'MacAcceleratorModelError'):
  p.remove(key)
print('live shadow configured:', p.get('MacAcceleratorHost'),
      'deadline_ms=', p.get('MacAcceleratorDeadlineMs'))
PY"

echo "Live shadow is enabled for the next on-road modeld start."
echo "The local 3X model remains the only control source."
