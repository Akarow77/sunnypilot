import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pyray as rl

from openpilot.cereal import log
from openpilot.selfdrive.ui.mici.onroad import driver_state as four_dm
from openpilot.selfdrive.ui.onroad import driver_state as stock_three_dm
from openpilot.selfdrive.ui.sunnypilot.onroad import driver_state as three_dm
from openpilot.selfdrive.ui.ui_state import UIStatus


class FakeSubMaster(dict):
  def __init__(self):
    super().__init__({
      "driverMonitoringState": SimpleNamespace(
        activePolicy=log.DriverMonitoringState.MonitoringPolicy.vision,
        isRHD=False,
        visionPolicyState=SimpleNamespace(faceDetected=True, awarenessPercent=100,
                                         pose=SimpleNamespace(pitch=0.0, yaw=0.0)),
      ),
      "driverStateV2": SimpleNamespace(leftDriverData=None, rightDriverData=None),
      "selfdriveState": SimpleNamespace(alertSize=four_dm.AlertSize.none),
    })
    self.valid = dict.fromkeys(self, True)
    self.alive = dict.fromkeys(self, True)
    self.recv_frame = dict.fromkeys(self, 101)
    self.recv_time = dict.fromkeys(self, 1000.0)


class TestCommaFourDMOnThreeX(unittest.TestCase):
  def setUp(self):
    self.sm = FakeSubMaster()
    self.ui = SimpleNamespace(sm=self.sm, started=True, started_frame=100,
                              status=UIStatus.LAT_ONLY, always_on_dm=False, model_confidence=False, rocket_fuel=False)
    for module in (four_dm, three_dm, stock_three_dm):
      patcher = patch.object(module, "ui_state", self.ui)
      patcher.start()
      self.addCleanup(patcher.stop)
    texture = patch.object(four_dm.gui_app, "texture", side_effect=lambda path, w, h: SimpleNamespace(width=w, height=h))
    texture.start()
    self.addCleanup(texture.stop)
    self.bottom_offset = patch.object(three_dm, "get_bottom_dev_ui_offset", return_value=0).start()
    self.addCleanup(patch.stopall)
    self.clock = patch.object(three_dm.time, "monotonic", return_value=1000.0).start()
    self.draw_plain = patch.object(rl, "draw_texture_ex").start()
    self.draw_rotated = patch.object(rl, "draw_texture_pro").start()
    self.renderer = three_dm.DriverStateRendererSP()
    self.rect = rl.Rectangle(30, 30, 2100, 1020)

  def frame(self):
    self.renderer._update_state()
    self.renderer._render(self.rect)

  def test_matches_original_four_pose_attention_and_fade(self):
    original = four_dm.DriverStateRenderer()
    original.set_should_draw(True)
    vision = self.sm["driverMonitoringState"].visionPolicyState
    for pitch, yaw, awareness, rhd in [(0, 0, 100, False), (0.1, -0.2, 94, False),
                                        (-0.2, 0.3, 60, False), (0.05, 0.2, 100, True)] * 5:
      vision.pose.pitch, vision.pose.yaw, vision.awarenessPercent = pitch, yaw, awareness
      self.sm["driverMonitoringState"].isRHD = rhd
      self.frame()
      original._update_state()
      original._render(original.rect)
      actual = self.renderer._driver_state_renderer
      for name in ("_pitch_filter", "_yaw_filter", "_rotation_filter", "_fade_filter", "_color_fade_filter"):
        self.assertEqual(getattr(actual, name).x, getattr(original, name).x, name)
      self.assertEqual(actual.is_rhd, original.is_rhd)
      self.assertEqual(actual._awareness_unfull, original._awareness_unfull)

  def test_green_orange_cone_and_wheeltouch_fade(self):
    for _ in range(80):
      self.frame()
    color = self.draw_rotated.call_args.args[-1]
    self.assertEqual((color.r, color.g, color.b), four_dm.CONE_COLOR_GREEN)
    self.sm["driverMonitoringState"].visionPolicyState.awarenessPercent = 94
    for _ in range(80):
      self.frame()
    color = self.draw_rotated.call_args.args[-1]
    self.assertEqual((color.r, color.g, color.b), four_dm.CONE_COLOR_ORANGE)
    self.sm["driverMonitoringState"].activePolicy = log.DriverMonitoringState.MonitoringPolicy.wheeltouch
    self.draw_rotated.reset_mock()
    for _ in range(80):
      self.frame()
    self.draw_rotated.assert_not_called()
    self.assertAlmostEqual(self.renderer._driver_state_renderer._fade_filter.x, 0.35, places=6)

  def test_missing_stale_and_previous_drive_messages_clear_display(self):
    for service in three_dm.DM_SERVICES:
      for field, value in (("valid", False), ("alive", False), ("recv_frame", 100), ("recv_time", 999.49), ("recv_time", 1000.01)):
        with self.subTest(service=service, field=field):
          self.frame()
          values = getattr(self.sm, field)
          before = values[service]
          values[service] = value
          self.draw_plain.reset_mock()
          self.draw_rotated.reset_mock()
          self.frame()
          self.assertFalse(self.renderer._data_valid)
          self.assertEqual(self.renderer._driver_state_renderer._fade_filter.x, 0.0)
          self.draw_plain.assert_not_called()
          self.draw_rotated.assert_not_called()
          values[service] = before
    self.ui.started = False
    self.frame()
    self.assertFalse(self.renderer._data_valid)

  def test_new_drive_resets_pose_even_without_offroad_render(self):
    self.sm["driverMonitoringState"].visionPolicyState.pose.yaw = 0.4
    for _ in range(20):
      self.frame()
    self.ui.started_frame = 200
    self.sm.recv_frame = dict.fromkeys(self.sm, 201)
    self.sm["driverMonitoringState"].visionPolicyState.pose.yaw = 0.0
    self.frame()
    self.assertAlmostEqual(self.renderer._driver_state_renderer._rotation_filter.x, 90.0)

  def test_malformed_awareness_is_hidden(self):
    for value in (-1, 101, math.nan):
      self.sm["driverMonitoringState"].visionPolicyState.awarenessPercent = value
      self.frame()
      self.assertFalse(self.renderer._data_valid)

  def test_nonfinite_pose_recovers_without_poisoned_filters(self):
    vision = self.sm["driverMonitoringState"].visionPolicyState
    for field in ("pitch", "yaw"):
      for bad in (math.nan, math.inf, -math.inf):
        with self.subTest(field=field, bad=bad):
          setattr(vision.pose, field, bad)
          self.frame()
          self.assertFalse(self.renderer._data_valid)
          setattr(vision.pose, field, 0.0)
          self.frame()
          self.assertTrue(self.renderer._data_valid)
          actual = self.renderer._driver_state_renderer
          self.assertTrue(math.isfinite(actual._rotation_filter.x))
          self.assertAlmostEqual(actual._rotation_filter.x, 90.0)

  def test_disengaged_and_always_on_dm_match_four_visibility(self):
    self.ui.status = UIStatus.DISENGAGED
    self.frame()
    self.assertEqual(self.renderer._driver_state_renderer._fade_filter.x, 0.0)
    self.ui.always_on_dm = True
    self.frame()
    self.assertGreater(self.renderer._driver_state_renderer._fade_filter.x, 0.0)

  def test_alert_visibility_matches_stock_three_x_and_recovers(self):
    stock = stock_three_dm.DriverStateRenderer()
    for size in (four_dm.AlertSize.small, four_dm.AlertSize.mid, four_dm.AlertSize.full):
      with self.subTest(alert_size=size):
        self.sm["selfdriveState"].alertSize = four_dm.AlertSize.none
        for _ in range(20):
          self.frame()
        self.assertTrue(stock.is_visible)
        self.assertGreater(self.renderer._driver_state_renderer._fade_filter.x, 0.0)

        self.sm["selfdriveState"].alertSize = size
        self.draw_plain.reset_mock()
        self.draw_rotated.reset_mock()
        self.frame()
        self.assertFalse(stock.is_visible)
        self.assertEqual(self.renderer._driver_state_renderer._fade_filter.x, 0.0)
        self.draw_plain.assert_not_called()
        self.draw_rotated.assert_not_called()

        self.sm["selfdriveState"].alertSize = four_dm.AlertSize.none
        self.frame()
        self.assertTrue(stock.is_visible)
        self.assertGreater(self.draw_plain.call_args.args[-1].a, 0)

  def test_rhd_and_developer_ui_avoid_confidence_panel(self):
    self.frame()
    actual = self.renderer._driver_state_renderer
    self.assertEqual(actual.rect.x, 60)
    self.assertEqual(actual.rect.y, 828)
    self.assertEqual(actual.rect.width, three_dm.DM_ICON_SIZE)
    self.sm["driverMonitoringState"].isRHD = True
    self.ui.model_confidence = True
    self.bottom_offset.return_value = 60
    self.frame()
    self.assertEqual(actual.rect.x, 30 + 2100 - three_dm.CONFIDENCE_PANEL_WIDTH - 30 - three_dm.DM_ICON_SIZE)
    self.assertEqual(actual.rect.y, 768)

  def test_lhd_dm_leaves_gap_for_acceleration_bar(self):
    self.ui.rocket_fuel = True
    self.ui.model_confidence = True
    self.frame()
    actual = self.renderer._driver_state_renderer
    self.assertGreaterEqual(actual.rect.x, self.rect.x + three_dm.RocketFuel.BAR_WIDTH + 16)
    self.assertLess(actual.rect.x + actual.rect.width, self.rect.x + self.rect.width / 2)
    self.assertEqual(actual.rect.y, 828)


if __name__ == "__main__":
  unittest.main()
