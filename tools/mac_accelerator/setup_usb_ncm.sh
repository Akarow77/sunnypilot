#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${ADB:-}" ]]; then
  :
elif command -v adb >/dev/null 2>&1; then
  ADB="$(command -v adb)"
elif [[ -x /opt/homebrew/bin/adb ]]; then
  ADB=/opt/homebrew/bin/adb
elif [[ -x /usr/local/bin/adb ]]; then
  ADB=/usr/local/bin/adb
else
  ADB=adb
fi
USB_DEVICE_IFACE="${USB_DEVICE_IFACE:-usb0}"

if ! "$ADB" get-state >/dev/null 2>&1; then
  echo "No authorized comma device found over ADB." >&2
  exit 1
fi

"$ADB" shell "ip link set '$USB_DEVICE_IFACE' up"
for _ in {1..20}; do
  MAC_USB_IFACE="$(networksetup -listallhardwareports | awk '/Hardware Port: Linux USB Gadget/{getline; print $2; exit}')"
  if [[ -n "$MAC_USB_IFACE" ]] && ifconfig "$MAC_USB_IFACE" | grep -q 'status: active'; then
    MAC_LINK_LOCAL="$(ifconfig "$MAC_USB_IFACE" | awk '/inet6 fe80:/{split($2, address, "%"); print address[1]; exit}')"
    DEVICE_LINK_LINE="$("$ADB" shell "ip -6 -o addr show dev '$USB_DEVICE_IFACE' scope link" | tr -d '\r')"
    if [[ -n "$DEVICE_LINK_LINE" ]] && [[ "$DEVICE_LINK_LINE" != *tentative* ]]; then
      DEVICE_LINK_LOCAL="$(printf '%s\n' "$DEVICE_LINK_LINE" | awk '{print $4}' | cut -d/ -f1)"
      echo "Mac:   ${MAC_LINK_LOCAL}%${MAC_USB_IFACE}"
      echo "comma: ${DEVICE_LINK_LOCAL}%${USB_DEVICE_IFACE}"
      exit 0
    fi
  fi
  sleep 0.25
done

echo "USB NCM did not become active." >&2
exit 1
