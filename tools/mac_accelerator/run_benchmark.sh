#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"
DEV=METAL JIT=2 exec "$REPO_ROOT/.venv/bin/python" "$SCRIPT_DIR/benchmark_model.py" "$@"
