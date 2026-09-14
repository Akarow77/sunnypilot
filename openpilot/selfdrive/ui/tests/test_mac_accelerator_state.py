#!/usr/bin/env python3

import unittest

from openpilot.selfdrive.ui.mac_accelerator_state import mac_accelerator_state


class MacAcceleratorStateTest(unittest.TestCase):
  def test_disconnected(self) -> None:
    self.assertIsNone(mac_accelerator_state(present=False, loading=False, ready=False, active=False, failed=False))

  def test_loading_ready_active_sequence(self) -> None:
    self.assertEqual(mac_accelerator_state(present=True, loading=True, ready=False, active=False, failed=False), 'loading')
    self.assertEqual(mac_accelerator_state(present=True, loading=False, ready=True, active=False, failed=False), 'ready')
    self.assertEqual(mac_accelerator_state(present=True, loading=False, ready=True, active=True, failed=False), 'active')

  def test_failure_has_priority_and_latches_in_caller(self) -> None:
    self.assertEqual(mac_accelerator_state(present=True, loading=True, ready=True, active=True, failed=True), 'failed')


if __name__ == '__main__':
  unittest.main()
