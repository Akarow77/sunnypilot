import math
import unittest

from openpilot.selfdrive.ui.mici.onroad.torque_bar_geometry import TORQUE_ANGLE_SPAN, torque_angle_span


class TestTorqueBarGeometry(unittest.TestCase):
  def test_mici_default_size_unchanged(self):
    self.assertEqual(torque_angle_span(476, 1), TORQUE_ANGLE_SPAN)

  def test_3x_restores_pre_navigation_span(self):
    # Full camera view and the narrower view with the sidebar open.
    for width in (2160, 1920, 1860):
      with self.subTest(width=width):
        self.assertEqual(torque_angle_span(width, 3), TORQUE_ANGLE_SPAN)

  def test_navigation_still_shrinks_arc(self):
    self.assertAlmostEqual(torque_angle_span(238, 1), TORQUE_ANGLE_SPAN / 2)

  def test_scaling_does_not_apply_twice(self):
    for width in (238, 476, 720):
      with self.subTest(width=width):
        self.assertAlmostEqual(torque_angle_span(width, 1), torque_angle_span(width * 3, 3))

  def test_larger_view_never_increases_angle(self):
    for scale in (1, 2, 3):
      for width in (100, 238, 476, 1428, 2160, 3840):
        with self.subTest(scale=scale, width=width):
          self.assertLessEqual(torque_angle_span(width, scale), TORQUE_ANGLE_SPAN)

  def test_3x_arc_chord_restores_original_size(self):
    # Outer arc chord at full torque; rounded end caps are additional.
    outer_radius = (1200 + 56) * 3
    width = 2 * outer_radius * math.sin(math.radians(torque_angle_span(2160, 3)) / 2)
    self.assertGreater(width, 800)
    self.assertLess(width, 850)

  def test_empty_view_has_no_angular_span(self):
    for width in (0, -1):
      self.assertEqual(torque_angle_span(width, 3), 0)
