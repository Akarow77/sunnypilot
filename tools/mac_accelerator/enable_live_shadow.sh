#!/usr/bin/env bash
set -euo pipefail

echo "Live shadow is unavailable: synchronous QCOM capture delayed the local model." >&2
echo "The production modeld capture hook has been removed pending an isolated capture path." >&2
echo "No device settings were changed. Use local_shadow_benchmark.py for Mac-only tests." >&2
exit 1
