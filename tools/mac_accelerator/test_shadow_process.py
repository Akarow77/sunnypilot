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
  def test_uses_low_realtime_priority_on_dedicated_little_core(self, _cpu_count, _sched_param, scheduler, affinity):
    self.assertEqual(shadow_process.configure_shadow_scheduling(), 'fifo-5-core-2')
    scheduler.assert_called_once_with(0, shadow_process.SCHED_FIFO, 5)
    affinity.assert_called_once_with(0, [2])

  @patch('shadow_process.sys.platform', 'linux')
  @patch('shadow_process.os.sched_setaffinity', create=True)
  @patch('shadow_process.os.sched_setscheduler', create=True)
  @patch('shadow_process.os.sched_param', side_effect=lambda value: value, create=True)
  @patch('shadow_process.os.cpu_count', return_value=8)
  def test_falls_back_to_little_core_timesharing(self, _cpu_count, _sched_param, scheduler, affinity):
    scheduler.side_effect = [PermissionError('denied'), None]
    self.assertEqual(shadow_process.configure_shadow_scheduling(), 'timeshare-little-cores')
    self.assertEqual(scheduler.call_args_list[1].args[:2], (0, shadow_process.SCHED_OTHER))
    affinity.assert_called_once_with(0, [0, 1, 2, 3])

  @patch('shadow_process.sys.platform', 'linux')
  @patch('shadow_process.os.sched_setaffinity', side_effect=PermissionError('denied'), create=True)
  @patch('shadow_process.os.sched_setscheduler', side_effect=PermissionError('denied'), create=True)
  @patch('shadow_process.os.sched_param', side_effect=lambda value: value, create=True)
  @patch('shadow_process.os.cpu_count', return_value=8)
  def test_scheduling_failure_never_blocks_shadow_startup(self, _cpu_count, _sched_param, _scheduler, _affinity):
    self.assertEqual(shadow_process.configure_shadow_scheduling(), 'timeshare-little-cores')


if __name__ == '__main__':
  unittest.main()
