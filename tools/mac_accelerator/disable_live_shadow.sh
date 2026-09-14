#!/usr/bin/env bash
set -euo pipefail

if [[ "$(adb shell 'cat /data/params/d/IsOffroad 2>/dev/null' | tr -d '\r')" != "1" ]]; then
  echo "Refusing to change live-shadow configuration while the comma 3X is on-road." >&2
  exit 1
fi

adb shell "cd /data/openpilot && exec sudo -u comma -- env \
  PATH=/usr/comma/shims:/usr/local/venv/bin:/usr/local/bin:/usr/bin:/bin \
  PYTHONPATH=/data/openpilot python3 - <<'PY'
from openpilot.common.params import Params

p = Params()
p.put_bool('MacAcceleratorShadowEnabled', False, block=True)
for key in ('MacAcceleratorPresent', 'MacAcceleratorLoading', 'MacAcceleratorReady',
            'MacAcceleratorActive', 'MacAcceleratorModelError'):
  p.remove(key)
print('live shadow disabled')
PY"
