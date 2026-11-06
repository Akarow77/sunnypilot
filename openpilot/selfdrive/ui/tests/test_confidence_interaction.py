import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pyray as rl

from openpilot.cereal import messaging
from openpilot.common.params import Params
from openpilot.selfdrive.ui import UI_BORDER_SIZE
from openpilot.selfdrive.ui.layouts import main
from openpilot.selfdrive.ui.onroad import augmented_road_view as road
from openpilot.selfdrive.ui.sunnypilot.layouts.settings import visuals
from openpilot.selfdrive.ui.sunnypilot.onroad import confidence, hud_renderer as hud
from openpilot.selfdrive.ui.sunnypilot.ui_state import UIStateSP
from openpilot.selfdrive.ui.ui_state import UIStatus
from openpilot.system.ui.lib.application import MouseEvent, MousePos, gui_app
from openpilot.system.ui.sunnypilot.widgets import list_view as sp_list, toggle as sp_toggle
from openpilot.system.ui import widgets


class EmptyOverlay(widgets.Widget):
  def _render(self, rect):
    pass


class TestConfidenceInteraction(unittest.TestCase):
  """Real Widget input, MainLayout/road geometry and isolated settings persistence.

  Camera IPC and GPU drawing are isolated; mouse dispatch, road click handlers,
  sidebar selection, confidence layout and the settings toggle remain real.
  """

  def setUp(self):
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    self.params = Params(directory.name)
    self.addCleanup(patch.stopall)
    self.events = SimpleNamespace(mouse_events=[], show_touches=False)
    patch.object(widgets, 'gui_app', self.events).start()
    patch.object(widgets, 'device', SimpleNamespace(awake=True)).start()
    patch.object(gui_app, 'font', return_value=rl.Font()).start()
    for name in ('begin_scissor_mode', 'end_scissor_mode', 'draw_rectangle_rec',
                 'draw_rectangle_gradient_v', 'draw_rectangle_rounded', 'draw_circle'):
      patch.object(rl, name).start()
    patch.object(rl, 'get_frame_time', return_value=0.05).start()
    self.draw_dot = patch.object(confidence, 'draw_circle_gradient').start()

    # The actual SP parameter reader needs no running UI, network or CarParams.
    self.state = object.__new__(UIStateSP)
    for name, value in dict(params=self.params, CP=None, CP_SP=None,
                            has_longitudinal_control=False, has_icbm=False,
                            chestnut_present=False, chestnut_active=False,
                            chestnut_loading=False, chestnut_compiled=False,
                            _sp_initialized=True, started=True, started_frame=10,
                            status=UIStatus.LAT_ONLY, is_body=False).items():
      setattr(self.state, name, value)
    self.state.is_offroad = lambda: False
    self.state.update_params()

    class Messages(dict):
      pass

    services = ('modelV2', 'selfdriveState', 'selfdriveStateSP')
    self.state.sm = Messages({s: getattr(messaging.new_message(s), s) for s in services})
    for name, value in (('valid', True), ('alive', True), ('updated', True),
                        ('recv_frame', 20), ('recv_time', time.monotonic())):
      setattr(self.state.sm, name, dict.fromkeys(services, value))
    predictions = self.state.sm['modelV2'].meta.disengagePredictions
    predictions.brakeDisengageProbs = [0.1]
    predictions.steerOverrideProbs = [0.2]
    for module in (main, road, confidence, hud, visuals):
      patch.object(module, 'ui_state', self.state).start()

    self.gauge = confidence.ModelConfidenceRenderer()
    self.gauge_click = Mock()
    self.gauge.set_click_callback(self.gauge_click)
    self.hud = object.__new__(hud.HudRendererSP)
    widgets.Widget.__init__(self.hud)
    self.hud._update_state = Mock()
    self.hud.is_cruise_available = False
    self.hud.speed_renderer = Mock()
    self.hud._model_confidence = self.gauge
    for name in ('developer_ui', 'road_name_renderer', 'speed_limit_renderer',
                 'smart_cruise_control_renderer', 'turn_signal_controller',
                 'circular_alerts_renderer'):
      setattr(self.hud, name, EmptyOverlay())
    self.hud.rocket_fuel = SimpleNamespace(render=Mock())

    # Avoid camera/resource construction while retaining the road's actual
    # render/layout and press/release methods through Widget.render().
    self.road = object.__new__(road.AugmentedRoadView)
    widgets.Widget.__init__(self.road)
    self.road.close = Mock()
    self.road._switch_stream_if_needed = Mock()
    self.road._update_calibration = Mock()
    self.road._draw_border = Mock()
    self.road.model_renderer = EmptyOverlay()
    self.road._hud_renderer = self.hud
    self.road.alert_renderer = EmptyOverlay()
    self.road.driver_state_renderer = EmptyOverlay()
    patch.object(road.CameraView, '_render').start()
    patch.object(road.AugmentedRoadViewSP, 'update_fade_out_bottom_overlay').start()

    self.main = object.__new__(main.MainLayout)
    widgets.Widget.__init__(self.main)
    self.main._sidebar = EmptyOverlay()  # Stock Sidebar starts visible.
    self.main._current_mode = main.MainState.ONROAD
    self.main._prev_onroad = True
    self.main._layouts = {main.MainState.ONROAD: self.road}
    self.main.set_rect(rl.Rectangle(0, 0, 2160, 1080))
    self.main._sidebar.set_visible(False)
    self.road.set_click_callback(self.main._on_onroad_clicked)

  def frame(self, event=None):
    self.events.mouse_events = [] if event is None else [event]
    for service in self.state.sm:
      self.state.sm.recv_time[service] = time.monotonic()
    self.main.render(self.main.rect)

  def touch(self, x, y, pressed):
    return MouseEvent(MousePos(x, y), 0, pressed, not pressed, pressed, time.monotonic())

  def test_touch_on_passive_gauge_toggles_sidebar_once_per_press(self):
    self.params.put_bool('ModelConfidence', True, block=True)
    self.state.update_params()
    self.frame()
    panel = self.gauge.rect
    x, y = panel.x + panel.width / 2, panel.y + panel.height / 2

    self.frame(self.touch(x, y, True))
    self.assertTrue(self.main._sidebar.is_visible)
    self.frame(self.touch(x, y, False))
    self.assertTrue(self.main._sidebar.is_visible)  # Release must not toggle twice.
    self.assertEqual(self.road.rect.x, main.SIDEBAR_WIDTH)
    self.assertEqual(self.hud.rect.x, main.SIDEBAR_WIDTH + UI_BORDER_SIZE)
    self.assertEqual(self.hud.rect.width, 2160 - main.SIDEBAR_WIDTH - 2 * UI_BORDER_SIZE)
    speed_rect = self.hud.speed_renderer.render.call_args.args[0]
    self.assertEqual(speed_rect.x + speed_rect.width / 2, (2160 + main.SIDEBAR_WIDTH) / 2)

    self.frame(self.touch(x, y, True))
    self.assertFalse(self.main._sidebar.is_visible)
    self.frame(self.touch(x, y, False))
    self.assertFalse(self.main._sidebar.is_visible)
    self.assertEqual(self.road.rect.x, 0)
    speed_rect = self.hud.speed_renderer.render.call_args.args[0]
    self.assertEqual(speed_rect.x + speed_rect.width / 2, 1080)
    self.gauge_click.assert_not_called()
    self.assertFalse(self.gauge.is_pressed)

  def test_hidden_gauge_and_sidebar_area_do_not_capture_road_touches(self):
    self.frame()
    self.assertFalse(self.state.model_confidence)
    self.draw_dot.assert_not_called()
    self.frame(self.touch(2070, 540, True))
    self.frame(self.touch(2070, 540, False))
    self.assertTrue(self.main._sidebar.is_visible)
    self.frame(self.touch(100, 540, True))
    self.frame(self.touch(100, 540, False))
    self.assertTrue(self.main._sidebar.is_visible)
    self.gauge_click.assert_not_called()

  def test_reopen_sidebar_after_viewport_resize_while_closed(self):
    self.params.put_bool('ModelConfidence', True, block=True)
    self.state.update_params()
    self.frame()
    # A closed-sidebar resize used to replace the cached road area with the
    # full viewport. Opening it later at unchanged bounds must recompute it.
    self.main.set_rect(rl.Rectangle(0, 0, 2200, 1080))
    self.main.set_rect(rl.Rectangle(0, 0, 2160, 1080))
    self.frame()
    self.assertFalse(self.main._sidebar.is_visible)
    self.assertEqual(self.road.rect.width, 2160)

    self.frame(self.touch(2070, 540, True))
    self.frame(self.touch(2070, 540, False))
    self.assertTrue(self.main._sidebar.is_visible)
    self.assertEqual(self.road.rect.x, main.SIDEBAR_WIDTH)
    self.assertEqual(self.road.rect.width, 2160 - main.SIDEBAR_WIDTH)
    self.assertEqual(self.hud.rect.x, main.SIDEBAR_WIDTH + UI_BORDER_SIZE)
    speed_rect = self.hud.speed_renderer.render.call_args.args[0]
    self.assertEqual(speed_rect.x + speed_rect.width / 2, (2160 + main.SIDEBAR_WIDTH) / 2)
    self.assertEqual(self.gauge.rect.x + self.gauge.rect.width, 2160 - UI_BORDER_SIZE)
    self.gauge_click.assert_not_called()

  def test_visuals_toggle_persists_and_refreshes_actual_panel(self):
    for module in (visuals, sp_toggle, sp_list):
      patch.object(module, 'Params', return_value=self.params).start()
    settings = visuals.VisualsLayout()
    toggle = settings._toggles['ModelConfidence'].action_item.toggle
    toggle_rect = rl.Rectangle(100, 100, 200, 100)
    self.assertFalse(toggle.get_state())
    self.assertFalse(self.params.get_bool('ModelConfidence'))

    for expected in (True, False):
      for pressed in (True, False):
        self.events.mouse_events = [self.touch(120, 120, pressed)]
        toggle.render(toggle_rect)
      deadline = time.monotonic() + 1.0
      while self.params.get_bool('ModelConfidence') != expected and time.monotonic() < deadline:
        time.sleep(0.001)
      self.assertEqual(self.params.get_bool('ModelConfidence'), expected)
      self.state.update_params()
      self.assertEqual(self.state.model_confidence, expected)
      self.draw_dot.reset_mock()
      self.frame()
      self.assertEqual(self.draw_dot.called, expected)
      self.assertEqual(self.gauge._has_value, expected)
      settings._update_state()
      self.assertEqual(toggle.get_state(), expected)


if __name__ == '__main__':
  unittest.main()
