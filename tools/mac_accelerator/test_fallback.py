#!/usr/bin/env python3

import unittest

from fallback import AcceleratorState, FallbackLatch


class FallbackLatchTest(unittest.TestCase):
  def test_deadline_failure_is_latched(self) -> None:
    latch = FallbackLatch(50.0)
    latch.activate()
    latch.observe(10, 50.1)
    latch.observe(11, 10.0)
    self.assertEqual(latch.state, AcceleratorState.FAILED)
    self.assertEqual(latch.failure_frame, 10)

  def test_invalid_output_fails(self) -> None:
    latch = FallbackLatch(50.0)
    latch.activate()
    latch.observe(4, 12.0, valid=False)
    self.assertEqual(latch.failure_reason, 'invalid output')

  def test_reset_requires_reactivation(self) -> None:
    latch = FallbackLatch(50.0)
    latch.activate()
    latch.observe(1, 80.0)
    latch.reset()
    latch.observe(2, 80.0)
    self.assertEqual(latch.state, AcceleratorState.LOADING)
    self.assertIsNone(latch.failure_frame)


if __name__ == '__main__':
  unittest.main()
