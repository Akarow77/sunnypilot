import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pyray as rl

from openpilot.cereal import log, messaging
from openpilot.selfdrive.ui import UI_BORDER_SIZE
# Import the road view first, matching the real UI's SP extension import order.
from openpilot.selfdrive.ui.onroad import augmented_road_view as road
from openpilot.selfdrive.ui.onroad import alert_renderer as stock_alert
from openpilot.selfdrive.ui.sunnypilot.onroad import alert_renderer as alerts
from openpilot.selfdrive.ui.sunnypilot.onroad import confidence, hud_renderer as hud
from openpilot.selfdrive.ui.sunnypilot.onroad import speed_renderer as speed
from openpilot.selfdrive.ui.sunnypilot.onroad.developer_ui import DeveloperUiState
from openpilot.selfdrive.ui.ui_state import UIStatus

wrap_module = importlib.import_module('openpilot.system.ui.lib.wrap_text')
AlertSize = log.SelfdriveState.AlertSize


def bounds(rect):
  return (rect.x, rect.y, rect.width, rect.height)


class TestConfidenceRoadLayout(unittest.TestCase):
  """Exercise actual road/HUD/alert layout with camera and GPU drawing isolated."""

  def setUp(self):
    services = ('modelV2', 'selfdriveState', 'selfdriveStateSP')

    class SubMaster(dict):
      pass

    self.sm = SubMaster({s: getattr(messaging.new_message(s), s) for s in services})
    for field, value in (('valid', True), ('alive', True), ('updated', True),
                         ('recv_frame', 20), ('recv_time', 10.0)):
      setattr(self.sm, field, dict.fromkeys(services, value))
    self.sm['modelV2'].meta.disengagePredictions.brakeDisengageProbs = [0.9]
    self.sm['modelV2'].meta.disengagePredictions.steerOverrideProbs = [0.9]
    self.state = SimpleNamespace(sm=self.sm, started=True, started_frame=10, started_time=0.0,
                                 status=UIStatus.LAT_ONLY, model_confidence=True, torque_bar=False,
                                 developer_ui=DeveloperUiState.BOTH,
                                 onroad_brightness_handle_alerts=Mock())
    self.addCleanup(patch.stopall)
    for module in (road, stock_alert, alerts, confidence, hud):
      patch.object(module, 'ui_state', self.state).start()
    patch.object(confidence.time, 'monotonic', return_value=10.0).start()
    patch.object(road.gui_app, 'sunnypilot_ui', return_value=True).start()
    patch.object(road.gui_app, 'font', return_value=rl.Font()).start()
    # Keep the real wrap algorithm and height calculation, replacing only font metrics.
    def measure(font, text, size, spacing=0):
      lines = text.split('\n')
      return rl.Vector2(max(map(len, lines), default=0) * size * 0.45, len(lines) * size)
    self.measure = measure
    patch.object(alerts, 'measure_text_cached', side_effect=measure).start()
    patch.object(wrap_module, 'measure_text_cached', side_effect=measure).start()
    patch.object(wrap_module, 'font_fallback', side_effect=lambda font: font).start()
    patch.object(wrap_module, '_cache', {}).start()
    patch.object(rl, 'begin_scissor_mode').start()
    patch.object(rl, 'end_scissor_mode').start()
    patch.object(rl, 'draw_rectangle_rec').start()
    patch.object(rl, 'draw_rectangle_gradient_v').start()
    self.draw_circle = patch.object(confidence, 'draw_circle_gradient').start()
    patch.object(road.CameraView, '_render').start()
    patch.object(road.AugmentedRoadViewSP, 'update_fade_out_bottom_overlay').start()
    patch.object(hud.HudRenderer, '_render').start()

    self.events = []
    self.confidence = confidence.ModelConfidenceRenderer()
    self.confidence._update_state()
    self.confidence.render = Mock(side_effect=self._render_confidence)
    self.hud = object.__new__(hud.HudRendererSP)
    self.hud._model_confidence = self.confidence
    for name in ('developer_ui', 'road_name_renderer', 'speed_limit_renderer', 'smart_cruise_control_renderer',
                 'turn_signal_controller', 'circular_alerts_renderer', 'rocket_fuel'):
      setattr(self.hud, name, SimpleNamespace(render=Mock()))
    self.hud.render = Mock(side_effect=self._render_hud)

    self.alert = alerts.AlertRendererSP()
    self.alert._draw_background = Mock(side_effect=lambda rect, alert: self.events.append('alert'))
    self.alert._draw_text = Mock()
    self.alert.render = Mock(side_effect=self.alert._render)
    self.view = object.__new__(road.AugmentedRoadView)
    self.view.close = Mock()  # No camera textures or IPC resources were created.
    self.view._switch_stream_if_needed = Mock()
    self.view._update_calibration = Mock()
    self.view._draw_border = Mock()
    self.view.model_renderer = SimpleNamespace(render=Mock())
    self.view._hud_renderer = self.hud
    self.view.alert_renderer = self.alert
    self.view.driver_state_renderer = SimpleNamespace(render=Mock(side_effect=lambda rect: self.events.append('dm')))
    self.screen = rl.Rectangle(0, 0, 2160, 1080)

  def _render_confidence(self, rect):
    self.events.append('confidence')
    self.confidence._render(rect)

  def _render_hud(self, rect):
    self.hud._rect = rect  # The real Widget.render records this before _render.
    self.hud._render(rect)

  def frame(self, size=AlertSize.none, text='Changing lanes', detail=''):
    self.sm['selfdriveState'].alertSize = size
    self.sm['selfdriveState'].alertText1 = text
    self.sm['selfdriveState'].alertText2 = detail
    self.events.clear()
    self.draw_circle.reset_mock()
    self.alert._draw_background.reset_mock()
    self.view._render(self.screen)
    return self.confidence.render.call_args.args[0]

  def test_full_height_panel_and_reserved_debug_width_in_every_mode(self):
    for mode in DeveloperUiState:
      with self.subTest(mode=mode):
        self.state.developer_ui = mode
        panel = self.frame()
        content = self.view._content_rect
        self.assertEqual(panel.y, content.y)
        self.assertEqual(panel.height, content.height)
        self.assertEqual(panel.width, confidence.CONFIDENCE_PANEL_WIDTH)
        debug = self.hud.developer_ui.render.call_args.args[0]
        self.assertEqual(debug.x + debug.width, panel.x)
        self.assertEqual(debug.height, content.height)
        self.assertEqual(content.height, self.screen.height - 2 * UI_BORDER_SIZE)

  def test_speed_and_unit_stay_centered_when_confidence_is_toggled(self):
    self.state.hide_v_ego_ui = False
    self.state.is_metric = True
    with patch.object(speed, 'ui_state', self.state), \
         patch.object(speed, 'measure_text_cached', return_value=rl.Vector2(200, 80)), \
         patch.object(rl, 'draw_text_ex') as draw:
      self.hud.speed_renderer = speed.SpeedRenderer()
      self.hud.speed_renderer.speed = 64
      # Include the narrower viewport used while the sidebar is open.
      for viewport in (rl.Rectangle(30, 30, 2100, 1020), rl.Rectangle(330, 30, 1800, 1020)):
        self.hud._rect = viewport
        for enabled in (False, True):
          self.state.model_confidence = enabled
          road_width = viewport.width - (confidence.CONFIDENCE_PANEL_WIDTH if enabled else 0)
          draw.reset_mock()
          self.hud._draw_current_speed(rl.Rectangle(viewport.x, viewport.y, road_width, viewport.height))
          self.assertEqual(draw.call_count, 2)
          self.assertEqual(draw.call_args_list[0].args[1], '64')
          for call in draw.call_args_list:
            self.assertEqual(call.args[2].x + 100, viewport.x + viewport.width / 2)

  def test_steering_arc_retains_stock_viewport_and_bottom_offset(self):
    self.state.torque_bar = True
    self.hud._torque_bar = SimpleNamespace(render=Mock())
    with patch('openpilot.selfdrive.ui.sunnypilot.onroad.developer_ui.ui_state', self.state):
      for screen in (rl.Rectangle(0, 0, 2160, 1080), rl.Rectangle(300, 0, 1860, 1080)):
        self.screen = screen
        for mode in DeveloperUiState:
          self.state.developer_ui = mode
          for enabled in (False, True):
            self.state.model_confidence = enabled
            self.frame()
            content = self.view._content_rect
            arc = self.hud._torque_bar.render.call_args.args[0]
            bottom = 60 if mode in (DeveloperUiState.BOTTOM, DeveloperUiState.BOTH) else 0
            self.assertEqual(bounds(arc), (content.x, content.y, content.width, content.height - bottom))
            self.assertEqual(arc.x + arc.width / 2, screen.x + screen.width / 2)

  def test_every_alert_stays_outside_panel_and_does_not_move_dot(self):
    for mode in DeveloperUiState:
      self.state.developer_ui = mode
      panel = self.frame()
      reference_dot = self.draw_circle.call_args.args[:3]
      for size in (AlertSize.small, AlertSize.mid, AlertSize.full):
        with self.subTest(mode=mode, size=size):
          self.frame(size, detail='Monitor surrounding traffic')
          alert_rect = self.alert._draw_background.call_args.args[0]
          margin = 0 if size == AlertSize.full else stock_alert.ALERT_MARGIN
          debug_margin = 230 if size != AlertSize.full and mode in (DeveloperUiState.RIGHT, DeveloperUiState.BOTH) else 0
          self.assertEqual(alert_rect.x + alert_rect.width, panel.x - margin - debug_margin)
          self.assertEqual(self.draw_circle.call_args.args[:3], reference_dot)
          self.assertEqual(self.events, ['confidence', 'alert', 'dm'])

  def test_alert_bottom_clearance_tracks_bottom_debug_row(self):
    for enabled in (False, True):
      self.state.model_confidence = enabled
      for mode in DeveloperUiState:
        self.state.developer_ui = mode
        for size in (AlertSize.small, AlertSize.mid, AlertSize.full):
          with self.subTest(confidence=enabled, mode=mode, size=size):
            self.frame(size, 'Pay Attention', 'Driver Distracted')
            alert_rect = self.alert._draw_background.call_args.args[0]
            content = self.view._content_rect
            if size == AlertSize.full:
              expected_gap = 0
            else:
              expected_gap = stock_alert.ALERT_MARGIN
              if mode in (DeveloperUiState.BOTTOM, DeveloperUiState.BOTH):
                expected_gap += 40
            self.assertEqual(content.y + content.height - alert_rect.y - alert_rect.height, expected_gap)

  def test_wrapped_message_uses_real_dynamic_height_without_panel_intrusion(self):
    for size in (AlertSize.small, AlertSize.mid):
      self.frame(size, 'Changing lanes', 'Monitor traffic')
      short = self.alert._draw_background.call_args.args[0]
      panel = self.frame(size, 'Changing lanes safely while monitoring surrounding traffic ' * 3,
                         'Please remain attentive to the road and nearby vehicles ' * 2)
      long = self.alert._draw_background.call_args.args[0]
      self.assertGreater(long.height, short.height)
      self.assertLess(long.y, short.y)
      self.assertEqual(long.x + long.width, panel.x - stock_alert.ALERT_MARGIN - 230)
      self.assertGreaterEqual(long.y, self.view._content_rect.y)
      self.assertLessEqual(long.y + long.height, self.view._content_rect.y + self.view._content_rect.height)
      current = self.alert.get_alert(self.sm)
      self.assertEqual(bounds(long), bounds(self.alert._get_alert_rect(self.view._content_rect, current.size)))

  def test_alert_removal_restores_unobstructed_layout(self):
    self.frame(AlertSize.mid, 'Changing lanes', 'Monitor traffic')
    panel = self.frame(AlertSize.none)
    self.alert._draw_background.assert_not_called()
    self.assertEqual(self.events, ['confidence', 'dm'])
    self.assertEqual(panel.height, self.view._content_rect.height)
    self.draw_circle.assert_called_once()

  def test_mid_explicit_newlines_keep_line_spacing_and_symmetric_padding(self):
    # Execute the real text renderer down to draw_text_ex. Whole-string metrics
    # deliberately grow with newlines, as native font measurement does, so they
    # cannot accidentally be used as the height of each individual output line.
    self.alert._draw_text = alerts.AlertRendererSP._draw_text.__get__(self.alert)
    cases = (
      ('Bookmark Saved', 'Device is not paired\nPair your device'),
      ('Model Selection Failed\nSelected model is unavailable',
       'Choose a model\nConnect to the internet\nTry again'),
    )
    with patch.object(rl, 'draw_text_ex') as draw:
      for title, detail in cases:
        for enabled in (False, True):
          self.state.model_confidence = enabled
          for mode in DeveloperUiState:
            self.state.developer_ui = mode
            with self.subTest(title=title, confidence=enabled, mode=mode):
              draw.reset_mock()
              self.frame(AlertSize.mid, title, detail)
              alert_rect = self.alert._draw_background.call_args.args[0]
              title_lines, detail_lines = title.split('\n'), detail.split('\n')
              lines = title_lines + detail_lines
              sizes = [stock_alert.ALERT_FONT_BIG] * len(title_lines) + [stock_alert.ALERT_FONT_SMALL] * len(detail_lines)
              self.assertEqual([call.args[1] for call in draw.call_args_list], lines)
              text_height = sum(sizes) + alerts.ALERT_LINE_SPACING
              padding = stock_alert.ALERT_PADDING - stock_alert.ALERT_MARGIN
              self.assertEqual(alert_rect.height, text_height + 2 * padding)
              y = alert_rect.y + padding
              for index, (call, line, size) in enumerate(zip(draw.call_args_list, lines, sizes, strict=True)):
                self.assertEqual(call.args[3], size)
                self.assertEqual(call.args[2].y, y)
                text_width = self.measure(call.args[0], line, size).x
                self.assertAlmostEqual(call.args[2].x + text_width / 2, alert_rect.x + alert_rect.width / 2, delta=0.001)
                y += size
                if index + 1 == len(title_lines):
                  y += alerts.ALERT_LINE_SPACING
              self.assertEqual(alert_rect.y + alert_rect.height - y, padding)

  def test_full_alert_keeps_entire_camera_area_and_confidence_panel_separate(self):
    panel = self.frame(AlertSize.full, 'TAKE CONTROL IMMEDIATELY', 'System Unresponsive')
    alert_rect = self.alert._draw_background.call_args.args[0]
    content = self.view._content_rect
    self.assertEqual(bounds(alert_rect), (content.x, content.y, content.width - panel.width, content.height))
    self.assertEqual(self.events, ['confidence', 'alert', 'dm'])
    self.draw_circle.assert_called_once()

  def test_setting_off_reserves_only_the_active_debug_column(self):
    self.state.model_confidence = False
    for mode in DeveloperUiState:
      self.state.developer_ui = mode
      self.frame(AlertSize.small)
      self.draw_circle.assert_not_called()
      content = self.view._content_rect
      self.assertEqual(bounds(self.hud.developer_ui.render.call_args.args[0]), bounds(content))
      alert_rect = self.alert._draw_background.call_args.args[0]
      column_width = 230 if mode in (DeveloperUiState.RIGHT, DeveloperUiState.BOTH) else 0
      self.assertEqual(alert_rect.width, content.width - 2 * stock_alert.ALERT_MARGIN - column_width)
    self.frame(AlertSize.full)
    self.assertEqual(bounds(self.alert._draw_background.call_args.args[0]), bounds(self.view._content_rect))


if __name__ == '__main__':
  unittest.main()
