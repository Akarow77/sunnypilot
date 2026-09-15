#!/usr/bin/env python3

import unittest
from unittest.mock import patch

import shadow_process


class ShadowSchedulingTests(unittest.TestCase):
  @patch('shadow_process.sys.platform', 'linux')
  @patch('shadow_process.os.sched_setaffinity', create=True)
  @patch('shadow_process.os.sched_setscheduler', create=True)
  @patch('shadow_process.os.sched_param', side_effect=lambda value: value, create=True)
  @patch('shadow_process.os.cpu_count', return_value=8)
  def test_demotes_to_normal_scheduling_on_little_cores(self, _cpu_count, _sched_param, scheduler, affinity):
    self.assertEqual(shadow_process.configure_shadow_scheduling(), 'timeshare-little-cores')
    scheduler.assert_called_once_with(0, shadow_process.SCHED_OTHER, 0)
    affinity.assert_called_once_with(0, [0, 1, 2, 3])

  @patch('shadow_process.sys.platform', 'linux')
  @patch('shadow_process.os.sched_setaffinity', create=True)
  @patch('shadow_process.os.sched_setscheduler', create=True)
  @patch('shadow_process.os.sched_param', side_effect=lambda value: value, create=True)
  @patch('shadow_process.os.cpu_count', return_value=8)
  def test_demotion_failure_stops_before_worker_import(self, _cpu_count, _sched_param, scheduler, affinity):
    scheduler.side_effect = PermissionError('denied')
    with patch.dict('sys.modules', {'live_shadow': None}), self.assertRaises(PermissionError):
      shadow_process.main()
    affinity.assert_not_called()

  @patch('shadow_process.sys.platform', 'linux')
  @patch('shadow_process.os.sched_setaffinity', side_effect=PermissionError('denied'), create=True)
  @patch('shadow_process.os.sched_setscheduler', create=True)
  @patch('shadow_process.os.sched_param', side_effect=lambda value: value, create=True)
  @patch('shadow_process.os.cpu_count', return_value=8)
  def test_affinity_failure_keeps_normal_scheduling(self, _cpu_count, _sched_param, scheduler, _affinity):
    self.assertEqual(shadow_process.configure_shadow_scheduling(), 'timeshare-default-affinity')
    scheduler.assert_called_once_with(0, shadow_process.SCHED_OTHER, 0)


if __name__ == '__main__':
  unittest.main()
