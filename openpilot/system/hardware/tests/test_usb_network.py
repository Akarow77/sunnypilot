import ipaddress
from types import SimpleNamespace

from openpilot.system.hardware import usb_network


def test_parse_address():
  assert str(usb_network.parse_address(" 10.47.0.1/24 ")) == "10.47.0.1/24"
  assert str(usb_network.parse_address("192.168.7.1/30")) == "192.168.7.1/30"
  for bad in ("10.47.0.0/24", "10.47.0.255/24", "8.8.8.8/24", "10.0.0.1/31", "10.0.0.1", "nope",
              "127.0.0.1/24", "169.254.10.1/24", "192.168.1.1/8", "::1/24"):
    assert usb_network.parse_address(bad) is None, bad


def test_peer_from_leases(tmp_path, monkeypatch):
  leases = tmp_path / "dnsmasq-eth0.leases"
  leases.write_text("100 aa:bb 10.47.0.20 mac *\n200 cc:dd 10.47.0.30 mac *\n300 ee:ff 192.168.43.5 phone *\n")
  monkeypatch.setattr(usb_network, "configured_address", lambda: ipaddress.IPv4Interface("10.47.0.1/24"))
  monkeypatch.setattr(usb_network, "DNSMASQ_LEASES", str(leases))
  monkeypatch.setattr(usb_network.glob, "glob", lambda _: [str(leases)])
  assert usb_network.peer_address() == "10.47.0.30"  # newest on our subnet, not the hotspot's


def test_dnsmasq_hands_out_no_gateway():
  args = usb_network.dnsmasq_args(ipaddress.IPv4Interface("10.47.0.1/24"))
  assert "--dhcp-range=10.47.0.2,10.47.0.254,255.255.255.0,12h" in args
  assert "--dhcp-option=3" in args and "--dhcp-option=6" in args  # no router, no DNS server
  assert "--port=0" in args
  assert "--dhcp-range=10.47.0.1,10.47.0.253,255.255.255.0,12h" in usb_network.dnsmasq_args(ipaddress.IPv4Interface("10.47.0.254/24"))


def test_gateway_shares_router_dns_and_nat():
  address = ipaddress.IPv4Interface("10.47.0.1/24")
  args = usb_network.dnsmasq_args(address, gateway=True)
  assert "--port=0" not in args and "--dhcp-option=3" not in args  # dnsmasq is the router and DNS
  rules = usb_network.nat_rules(address)
  assert ("nat", "MASQUERADE") in [(t, r[-1]) for t, r in rules]
  assert any("--clamp-mss-to-pmtu" in r for _, r in rules)  # LTE MTU
  assert all(usb_network.RULE_TAG in r for _, r in rules)


def test_network_configuration_does_not_run_on_mac(monkeypatch):
  monkeypatch.setattr(usb_network, 'COMMA_HARDWARE', False)
  def forbidden():
    raise AssertionError('must not reconfigure host networking')
  monkeypatch.setattr(usb_network, '_up', forbidden)
  monkeypatch.setattr(usb_network, '_down', forbidden)
  assert usb_network.up() is False
  usb_network.down()


def test_activate_only_available_managed_usb_ethernet():
  from jeepney.low_level import HeaderFields
  devices = {
    '/eth0': (1, 'eth0', True, 70),  # existing DHCP profile is still connecting
    '/wifi': (2, 'wlan0', True, 100),
    '/gadget': (1, 'usb0', True, 100),
    '/unmanaged': (1, 'enx123', False, 10),
    '/unplugged': (1, 'eth1', True, 20),
  }
  activated = []

  class Connection:
    def send_and_get_reply(self, message):
      member = message.header.fields[HeaderFields.member]
      if member == 'GetDevices':
        return SimpleNamespace(body=(list(devices),))
      if member == 'GetAll':
        kind, interface, managed, state = devices[message.header.fields[HeaderFields.path]]
        return SimpleNamespace(body=({'DeviceType': ('u', kind), 'Interface': ('s', interface),
                                     'Managed': ('b', managed), 'State': ('u', state)},))
      assert member == 'ActivateConnection'
      activated.append(message.body)
      return SimpleNamespace(body=('/active',))

  usb_network._activate_connection(Connection(), '/profile')
  assert activated == [('/profile', '/eth0', '/')]
