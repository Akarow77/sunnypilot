import subprocess
import unittest
from unittest.mock import Mock, patch

from usb_ncm_supervisor import Supervisor, endpoint_is_current, parse_link_info, parse_scoped_endpoint


class FakeProcess:
  def __init__(self, returncode: int | None):
    self.returncode = returncode
    self.terminated = False

  def poll(self):
    return -15 if self.terminated else self.returncode

  def terminate(self):
    self.terminated = True

  def wait(self, timeout=None):
    del timeout
    return self.poll()


class UsbNcmSupervisorTests(unittest.TestCase):
  def test_parse_link_info(self):
    mac, comma = parse_link_info(
      'Mac:   fe80::1%en7\ncomma: fe80::2%usb0\n')
    self.assertEqual(mac, 'fe80::1%en7')
    self.assertEqual(comma, 'fe80::2%usb0')

  def test_rejects_unscoped_or_non_link_local_endpoint(self):
    for endpoint in ('fe80::1', '::1%lo0', '192.0.2.1%en7'):
      with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
        parse_scoped_endpoint(endpoint)

  @patch('usb_ncm_supervisor.socket.if_nametoindex', return_value=26)
  @patch('usb_ncm_supervisor.subprocess.run')
  def test_current_endpoint_requires_index_address_and_active_link(self, run, _ifindex):
    run.return_value = subprocess.CompletedProcess([], 0,
      'en7: flags=UP\n\tinet6 fe80::1%en7 prefixlen 64\n\tstatus: active\n', '')
    self.assertTrue(endpoint_is_current('fe80::1%en7', 26))
    self.assertFalse(endpoint_is_current('fe80::1%en7', 24))

  @patch('usb_ncm_supervisor.socket.if_nametoindex', return_value=26)
  @patch('usb_ncm_supervisor.subprocess.run')
  def test_inactive_or_missing_address_is_not_current(self, run, _ifindex):
    for state in ('status: inactive\n', 'status: active\n'):
      with self.subTest(state=state):
        run.return_value = subprocess.CompletedProcess([], 0, state, '')
        self.assertFalse(endpoint_is_current('fe80::1%en7', 26))

  @patch('usb_ncm_supervisor.endpoint_is_current', return_value=False)
  @patch('usb_ncm_supervisor.subprocess.Popen')
  def test_link_change_restarts_but_server_failure_does_not(self, popen, _current):
    first, second = FakeProcess(None), FakeProcess(7)
    popen.side_effect = [first, second]
    supervisor = Supervisor(Mock(), ['server'], retry_seconds=1, poll_seconds=1)
    supervisor.discover = Mock(side_effect=[('fe80::1%en7', 26), ('fe80::1%en7', 27)])
    supervisor.wait_or_stop = Mock(return_value=False)
    self.assertEqual(supervisor.run(), 7)
    self.assertTrue(first.terminated)
    self.assertEqual(popen.call_count, 2)
    self.assertEqual(supervisor.discover.call_count, 2)


if __name__ == '__main__':
  unittest.main()
