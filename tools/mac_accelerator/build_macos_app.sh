#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR/macos_app"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/dist}"
APP="$OUTPUT_DIR/Sunnypilot Mac Accelerator.app"
CONTENTS="$APP/Contents"
ICON_SOURCE="$SOURCE_DIR/AppIcon.png"
ICONSET="$OUTPUT_DIR/AppIcon.iconset"

mkdir -p "$CONTENTS/MacOS" "$CONTENTS/Resources"
xcrun --sdk macosx clang -fobjc-arc -fmodules \
  -fmodules-cache-path="$OUTPUT_DIR/ModuleCache" -O2 -mmacosx-version-min=15.0 \
  "$SOURCE_DIR/MacAcceleratorApp.m" \
  -o "$CONTENTS/MacOS/SunnypilotMacAccelerator" \
  -framework Cocoa -framework Security
mkdir -p "$ICONSET"
sips -z 16 16 "$ICON_SOURCE" --out "$ICONSET/icon_16x16.png" >/dev/null
sips -z 32 32 "$ICON_SOURCE" --out "$ICONSET/icon_16x16@2x.png" >/dev/null
sips -z 32 32 "$ICON_SOURCE" --out "$ICONSET/icon_32x32.png" >/dev/null
sips -z 64 64 "$ICON_SOURCE" --out "$ICONSET/icon_32x32@2x.png" >/dev/null
sips -z 128 128 "$ICON_SOURCE" --out "$ICONSET/icon_128x128.png" >/dev/null
sips -z 256 256 "$ICON_SOURCE" --out "$ICONSET/icon_128x128@2x.png" >/dev/null
sips -z 256 256 "$ICON_SOURCE" --out "$ICONSET/icon_256x256.png" >/dev/null
sips -z 512 512 "$ICON_SOURCE" --out "$ICONSET/icon_256x256@2x.png" >/dev/null
sips -z 512 512 "$ICON_SOURCE" --out "$ICONSET/icon_512x512.png" >/dev/null
sips -z 1024 1024 "$ICON_SOURCE" --out "$ICONSET/icon_512x512@2x.png" >/dev/null
/usr/bin/python3 "$SOURCE_DIR/build_icns.py" "$ICONSET" "$CONTENTS/Resources/AppIcon.icns"
cp "$ICON_SOURCE" "$CONTENTS/Resources/AppIcon.png"
cp "$SOURCE_DIR/Info.plist" "$CONTENTS/Info.plist"
codesign --force --sign - "$APP"
printf 'Built %s\n' "$APP"
