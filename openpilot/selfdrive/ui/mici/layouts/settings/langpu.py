import threading
import time

from openpilot.selfdrive.ui.mici.widgets.button import BigButton, BigParamControl, GreyBigButton
from openpilot.selfdrive.ui.mici.widgets.dialog import BigDialog, BigInputDialog
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.hardware import usb_network
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets.scroller import NavScroller


class _OffroadParamControl(BigParamControl):
  """Persist before applying network changes, and reject a stale onroad click."""

  def __init__(self, text, param, toggle_callback=None):
    super().__init__(text, param)
    self._offroad_callback = toggle_callback
    self.set_enabled(lambda: ui_state.is_offroad())

  def _handle_mouse_release(self, mouse_pos):
    if not ui_state.is_offroad():
      return
    super()._handle_mouse_release(mouse_pos)
    if self._offroad_callback is not None:
      self._offroad_callback(self._checked)


class LanGpuLayoutMici(NavScroller):
  """Run the big driving model on a computer on the network (e.g. a Mac running Bamyanggang)."""

  def __init__(self):
    super().__init__()
    self._usb_peer_checked = 0.

    enabled_toggle = _OffroadParamControl("lan gpu", "GpuOnLanEnabled")

    def server_callback():
      if not ui_state.is_offroad():
        return
      def on_confirm(text: str):
        if not ui_state.is_offroad():
          return
        text = text.strip()
        if text:
          ui_state.params.put("GpuOnLanHost", text)
        else:
          ui_state.params.remove("GpuOnLanHost")
        self._usb_peer_checked = 0.
      gui_app.push_widget(BigInputDialog("server IP[:port], empty for USB...", ui_state.params.get("GpuOnLanHost") or "",
                                         minimum_length=0, confirm_callback=on_confirm))

    self._server_btn = BigButton("server", ui_state.params.get("GpuOnLanHost") or "USB Ethernet")
    self._server_btn.set_click_callback(server_callback)
    self._server_btn.set_enabled(lambda: ui_state.is_offroad())

    def usb_network_callback(checked: bool):
      if ui_state.is_offroad():
        threading.Thread(target=usb_network.up if checked else usb_network.down, daemon=True).start()

    usb_toggle = _OffroadParamControl("usb network", "UsbNetworkEnabled", toggle_callback=usb_network_callback)

    def usb_address_callback():
      if not ui_state.is_offroad():
        return
      def on_confirm(text: str):
        if not ui_state.is_offroad():
          return
        if (address := usb_network.parse_address(text)) is None:
          gui_app.push_widget(BigDialog("invalid address", "Use a private IPv4 address with a /8 to /30 subnet, e.g. 10.47.0.1/24."))
          return
        ui_state.params.put("UsbNetworkAddress", str(address), block=True)
        self._usb_address_btn.set_value(str(address))
        if ui_state.params.get_bool("UsbNetworkEnabled"):
          threading.Thread(target=usb_network.up, daemon=True).start()  # re-apply on the new subnet
      gui_app.push_widget(BigInputDialog("10.47.0.1/24...", str(usb_network.configured_address()),
                                         confirm_callback=on_confirm))

    self._usb_address_btn = BigButton("usb network\naddress", str(usb_network.configured_address()))
    self._usb_address_btn.set_click_callback(usb_address_callback)
    self._usb_address_btn.set_enabled(lambda: ui_state.is_offroad() and ui_state.params.get_bool("UsbNetworkEnabled"))

    def usb_gateway_callback(checked: bool):
      if ui_state.is_offroad() and ui_state.params.get_bool("UsbNetworkEnabled"):
        threading.Thread(target=usb_network.up, daemon=True).start()

    usb_gateway_toggle = _OffroadParamControl("share internet", "UsbNetworkGateway", toggle_callback=usb_gateway_callback)
    usb_gateway_toggle.set_enabled(lambda: ui_state.is_offroad() and ui_state.params.get_bool("UsbNetworkEnabled"))

    self._scroller.add_widgets([
      enabled_toggle,
      GreyBigButton("LAN GPU", "Experimental network inference. Changes apply on the next drive. On failure, the local model takes over."),
      self._server_btn,
      GreyBigButton("server address", "IP[:port], default port 8000. Leave empty to use the computer on the USB Ethernet link."),
      usb_toggle,
      self._usb_address_btn,
      usb_gateway_toggle,
      GreyBigButton("share internet", "Optional: route the computer through this device. Uses LTE data when on LTE. Off keeps the computer's own internet."),
    ])

  def _update_state(self):
    super()._update_state()
    # show which computer lan gpu will use over the USB Ethernet adapter
    now = time.monotonic()
    if now - self._usb_peer_checked > 2.:
      self._usb_peer_checked = now
      if host := ui_state.params.get("GpuOnLanHost"):
        self._server_btn.set_value(host)
      else:
        peer = usb_network.peer_address() if ui_state.params.get_bool("UsbNetworkEnabled") else None
        self._server_btn.set_value(f"USB · {peer}" if peer else "USB Ethernet")
