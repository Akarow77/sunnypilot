#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR/macos_app"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/dist}"
APP="$OUTPUT_DIR/Sunnypilot Mac Accelerator.app"
CONTENTS="$APP/Contents"

mkdir -p "$CONTENTS/MacOS" "$CONTENTS/Resources"
xcrun --sdk macosx clang -fobjc-arc -fmodules \
  -fmodules-cache-path="$OUTPUT_DIR/ModuleCache" -O2 -mmacosx-version-min=15.0 \
  "$SOURCE_DIR/MacAcceleratorApp.m" \
  -o "$CONTENTS/MacOS/SunnypilotMacAccelerator" \
  -framework Cocoa -framework Security
cp "$SOURCE_DIR/Info.plist" "$CONTENTS/Info.plist"
codesign --force --sign - "$APP"
printf 'Built %s\n' "$APP"
