import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pyray as rl

from openpilot.cereal import messaging, log
from openpilot.selfdrive.ui.ui_state import UIStatus
from openpilot.selfdrive.ui.sunnypilot.ui_state import UIStateSP, MADSState, OpenpilotState
from openpilot.selfdrive.ui.sunnypilot.onroad import confidence


class TestConfidenceValue(unittest.TestCase):
  def test_active_axes(self):
    self.assertAlmostEqual(confidence.model_confidence(UIStatus.ENGAGED, [0.2, 0.1], [0.3, 0.1]), 0.56)
    self.assertAlmostEqual(confidence.model_confidence(UIStatus.LAT_ONLY, [], [0.3]), 0.7)
    self.assertAlmostEqual(confidence.model_confidence(UIStatus.LONG_ONLY, [0.2], []), 0.8)
    self.assertAlmostEqual(confidence.model_confidence(UIStatus.OVERRIDE, [0.2], [0.3]), 0.56)

  def test_missing_and_invalid_predictions(self):
    for bad in ([], [math.nan], [math.inf], [-0.01], [1.01], [0.1, math.nan]):
      with self.subTest(bad=bad):
        self.assertIsNone(confidence.model_confidence(UIStatus.LAT_ONLY, [0], bad))
        self.assertIsNone(confidence.model_confidence(UIStatus.ENGAGED, bad, [0]))
    self.assertEqual(confidence.model_confidence(UIStatus.ENGAGED, [1], [0]), 0)

  def test_original_color_thresholds(self):
    for value, expected in ((0.0, (255, 0, 21)), (0.2, (255, 0, 21)), (0.21, (255, 200, 0)),
                            (0.5, (255, 200, 0)), (0.51, (0, 255, 204)), (1.0, (0, 255, 204))):
      with self.subTest(value=value):
        color = confidence.confidence_colors(value)[0]
        self.assertEqual((color.r, color.g, color.b), expected)
    color = confidence.confidence_colors(1, override=True)[0]
    self.assertEqual((color.r, color.g, color.b), (255, 255, 255))
    color = confidence.confidence_colors(1, override=True)[1]
    self.assertEqual((color.r, color.g, color.b, color.a), (82, 82, 82, 255))
    for value in (0.0, 0.3, 1.0):
      colors = confidence.confidence_colors(value, inactive=True)
      self.assertEqual([(c.r, c.g, c.b, c.a) for c in colors], [(50, 50, 50, 255), (13, 13, 13, 255)])
      for color in confidence.confidence_colors(value, inactive=True):
        self.assertEqual(color.r, color.g)
        self.assertEqual(color.g, color.b)
        self.assertGreater(color.a, 0)


class TestConfidenceWidget(unittest.TestCase):
  def setUp(self):
    services = ('modelV2', 'selfdriveState', 'selfdriveStateSP')
    messages = {s: getattr(messaging.new_message(s), s) for s in services}
    messages['modelV2'].meta.disengagePredictions.brakeDisengageProbs = [0.1] * 5
    messages['modelV2'].meta.disengagePredictions.steerOverrideProbs = [0.1] * 5

    class SubMaster(dict):
      pass

    self.sm = SubMaster(messages)
    self.sm.valid = dict.fromkeys(services, True)
    self.sm.alive = dict.fromkeys(services, True)
    self.sm.recv_frame = dict.fromkeys(services, 20)
    self.sm.recv_time = dict.fromkeys(services, 10.0)
    self.state = SimpleNamespace(sm=self.sm, started=True, started_frame=10, status=UIStatus.LAT_ONLY, model_confidence=True)
    self.addCleanup(patch.stopall)
    patch.object(confidence, 'ui_state', self.state).start()
    patch.object(confidence.time, 'monotonic', return_value=10.0).start()
    self.widget = confidence.ModelConfidenceRenderer()
    self.settle_frames = math.ceil(4.0 / self.widget._confidence_filter.dt)

  def set_mads_status(self, expected):
    """Use the production status mapping, including paused MADS -> override."""
    ss, ss_sp = self.sm['selfdriveState'], self.sm['selfdriveStateSP']
    ss.enabled = expected in (UIStatus.ENGAGED, UIStatus.OVERRIDE, UIStatus.LONG_ONLY)
    ss.state = OpenpilotState.enabled if ss.enabled else OpenpilotState.disabled
    ss_sp.mads.available = True
    ss_sp.mads.enabled = expected in (UIStatus.ENGAGED, UIStatus.OVERRIDE, UIStatus.LAT_ONLY)
    ss_sp.mads.state = MADSState.paused if expected == UIStatus.OVERRIDE else (
      MADSState.enabled if ss_sp.mads.enabled else MADSState.disabled)
    self.state.status = UIStatus(UIStateSP.update_status(ss, ss_sp, []))
    self.assertEqual(self.state.status, expected)

  def test_lateral_only_color_changes_with_value(self):
    for prediction, expected in ((0.1, (0, 255, 204)), (0.7, (255, 200, 0)), (0.9, (255, 0, 21))):
      self.widget._reset()
      self.sm['modelV2'].meta.disengagePredictions.steerOverrideProbs = [prediction] * 5
      self.widget._update_state()
      self.assertTrue(self.widget._has_value)
      color = confidence.confidence_colors(self.widget._confidence_filter.x)[0]
      self.assertEqual((color.r, color.g, color.b), expected)

  def test_stale_value_cleared_and_not_reused_on_recovery(self):
    self.widget._update_state()
    self.assertGreater(self.widget._confidence_filter.x, 0.8)
    self.sm.recv_time['modelV2'] = 9.49
    self.widget._update_state()
    self.assertFalse(self.widget._has_value)
    self.sm.recv_time['modelV2'] = 10.0
    self.sm['modelV2'].meta.disengagePredictions.steerOverrideProbs = [0.95]
    self.widget._update_state()
    self.assertAlmostEqual(self.widget._confidence_filter.x, 0.05, places=6)

  def test_invalid_dead_old_session_and_future_values_cleared(self):
    for service in self.sm:
      for attr, invalid in (('valid', False), ('alive', False), ('recv_frame', 10), ('recv_time', 11.0)):
        with self.subTest(service=service, field=attr):
          field = getattr(self.sm, attr)
          previous = field[service]
          self.widget._update_state()
          field[service] = invalid
          self.widget._update_state()
          self.assertFalse(self.widget._has_value)
          field[service] = previous

  def test_offroad_and_disabled_hidden(self):
    for attr, value in (('started', False), ('model_confidence', False)):
      with self.subTest(attr=attr):
        previous = getattr(self.state, attr)
        setattr(self.state, attr, value)
        self.widget._update_state()
        self.assertFalse(self.widget._has_value)
        with patch.object(rl, 'draw_rectangle_rec') as panel, patch.object(confidence, 'draw_circle_gradient') as dot:
          self.widget._render(rl.Rectangle(1840, 30, 120, 960))
          panel.assert_not_called()
          dot.assert_not_called()
        setattr(self.state, attr, previous)

  def test_disengaging_descends_to_visible_grey_bottom_despite_high_model_value(self):
    self.state.status = UIStatus.ENGAGED
    self.sm['modelV2'].meta.disengagePredictions.brakeDisengageProbs = [0.0]
    self.sm['modelV2'].meta.disengagePredictions.steerOverrideProbs = [0.0]
    self.widget._update_state()
    self.assertEqual(self.widget._confidence_filter.x, 1.0)
    rect = rl.Rectangle(1840, 30, 120, 960)
    values, centers = [], []
    self.state.status = UIStatus.DISENGAGED
    with patch.object(rl, 'draw_rectangle_rec'), patch.object(confidence, 'draw_circle_gradient') as dot:
      for _ in range(self.settle_frames):
        self.widget._update_state()
        self.widget._render(rect)
        self.assertTrue(self.widget._has_value)
        values.append(self.widget._confidence_filter.x)
        centers.append(dot.call_args.args[1])
        self.assertEqual([(c.r, c.g, c.b) for c in dot.call_args.args[-2:]], [(50, 50, 50), (13, 13, 13)])
      radius = dot.call_args.args[2]
      expected_bottom = rect.y + rect.height - math.ceil(radius * math.sqrt(2)) - 1
      self.assertEqual(centers[-1], expected_bottom)
      self.assertEqual(centers[-10:], [expected_bottom] * 10)
    self.assertGreater(values[0], 0.0)  # It moves down from the previous value, without jumping.
    self.assertLess(values[0], 1.0)
    self.assertTrue(all(a > b for a, b in zip(values, values[1:])))
    self.assertTrue(all(a <= b for a, b in zip(centers, centers[1:])))
    self.assertLess(values[-1], 0.0)
    # Match comma four's -0.5 target and 0.5-second time constant.
    dt = self.widget._confidence_filter.dt
    alpha = dt / (0.5 + dt)
    self.assertAlmostEqual(values[0], (1 - alpha) * 1.0 + alpha * -0.5)

  def test_reengagement_rises_gradually_from_disengaged_bottom(self):
    self.state.status = UIStatus.ENGAGED
    self.sm['modelV2'].meta.disengagePredictions.brakeDisengageProbs = [0.0]
    self.sm['modelV2'].meta.disengagePredictions.steerOverrideProbs = [0.0]
    self.widget._update_state()
    self.state.status = UIStatus.DISENGAGED
    for _ in range(self.settle_frames):
      self.widget._update_state()
    before = self.widget._confidence_filter.x
    self.assertLess(before, 0)
    self.state.status = UIStatus.ENGAGED
    values, centers = [], []
    with patch.object(rl, 'draw_rectangle_rec'), patch.object(confidence, 'draw_circle_gradient') as dot:
      for _ in range(self.settle_frames):
        self.widget._update_state()
        self.widget._render(rl.Rectangle(1840, 30, 120, 960))
        values.append(self.widget._confidence_filter.x)
        centers.append(dot.call_args.args[1])
    self.assertGreater(values[0], before)
    self.assertLess(values[0], 0)  # Preserve the original entrance animation, not a jump to 1.
    self.assertTrue(all(a < b for a, b in zip(values, values[1:])))
    self.assertTrue(all(a >= b for a, b in zip(centers, centers[1:])))
    self.assertGreater(values[-1], 0.9)

  def test_disengaged_descent_does_not_require_a_model_prediction(self):
    self.widget._update_state()
    before = self.widget._confidence_filter.x
    self.state.status = UIStatus.DISENGAGED
    self.sm.valid['modelV2'] = False
    self.sm.alive['modelV2'] = False
    self.sm.recv_time['modelV2'] = 0.0
    self.sm['modelV2'].meta.disengagePredictions.brakeDisengageProbs = []
    self.sm['modelV2'].meta.disengagePredictions.steerOverrideProbs = [math.nan]
    self.widget._update_state()
    self.assertTrue(self.widget._has_value)
    self.assertGreater(self.widget._confidence_filter.x, 0.0)
    self.assertLess(self.widget._confidence_filter.x, before)
    for _ in range(self.settle_frames):
      self.widget._update_state()
    self.assertLess(self.widget._confidence_filter.x, 0.0)
    self.state.status = UIStatus.ENGAGED
    self.widget._update_state()
    self.assertFalse(self.widget._has_value)  # Active display must still reject the missing model.

  def test_invalid_selfdrive_status_resets_disengaged_animation(self):
    for attr, invalid in (('valid', False), ('alive', False), ('recv_frame', 10),
                          ('recv_time', 9.49), ('recv_time', 11.0)):
      with self.subTest(attr=attr):
        self.state.status = UIStatus.LAT_ONLY
        self.widget._update_state()
        self.state.status = UIStatus.DISENGAGED
        self.widget._update_state()
        self.assertTrue(self.widget._has_value)
        field = getattr(self.sm, attr)
        previous = field['selfdriveState']
        field['selfdriveState'] = invalid
        self.widget._update_state()
        self.assertFalse(self.widget._has_value)
        self.assertFalse(self.widget._confidence_filter.initialized)
        self.assertEqual(self.widget._confidence_filter.x, 0.0)
        field['selfdriveState'] = previous
        self.widget._update_state()
        self.assertTrue(self.widget._has_value)
        self.assertEqual(self.widget._confidence_filter.x, -0.5)

  def test_offroad_and_new_drive_do_not_reuse_the_previous_animation(self):
    self.widget._update_state()
    self.assertGreater(self.widget._confidence_filter.x, 0.8)
    self.state.status = UIStatus.DISENGAGED
    self.widget._update_state()
    self.state.started = False
    self.widget._update_state()
    self.assertFalse(self.widget._has_value)
    self.assertFalse(self.widget._confidence_filter.initialized)
    self.state.started = True
    self.widget._update_state()
    self.assertEqual(self.widget._confidence_filter.x, -0.5)

    self.state.status = UIStatus.ENGAGED
    for _ in range(self.settle_frames):
      self.widget._update_state()
    self.assertGreater(self.widget._confidence_filter.x, 0.8)
    self.state.started_frame = 100
    self.sm.recv_frame = dict.fromkeys(self.sm, 101)
    self.state.status = UIStatus.DISENGAGED
    self.widget._update_state()
    self.assertTrue(self.widget._has_value)
    self.assertEqual(self.widget._confidence_filter.x, -0.5)

  def test_valid_override_retains_both_axis_formula_and_white_grey_palette(self):
    self.set_mads_status(UIStatus.OVERRIDE)
    self.sm['modelV2'].meta.disengagePredictions.brakeDisengageProbs = [0.2]
    self.sm['modelV2'].meta.disengagePredictions.steerOverrideProbs = [0.3]
    self.widget._update_state()
    self.assertTrue(self.widget._has_value)
    self.assertAlmostEqual(self.widget._confidence_filter.x, 0.56, places=6)
    with patch.object(rl, 'draw_rectangle_rec'), patch.object(confidence, 'draw_circle_gradient') as dot:
      self.widget._render(rl.Rectangle(1840, 30, 120, 960))
      self.assertEqual([(c.r, c.g, c.b, c.a) for c in dot.call_args.args[-2:]],
                       [(255, 255, 255, 255), (82, 82, 82, 255)])

  def test_engaged_override_transition_preserves_original_filter_at_20_and_60_fps(self):
    for fps in (20, 60):
      with self.subTest(fps=fps), patch.object(confidence.gui_app, '_target_fps', fps):
        self.widget = confidence.ModelConfidenceRenderer()
        self.assertAlmostEqual(self.widget._confidence_filter.dt, 1 / fps)
        self.set_mads_status(UIStatus.ENGAGED)
        predictions = self.sm['modelV2'].meta.disengagePredictions
        predictions.brakeDisengageProbs = [0.0]
        predictions.steerOverrideProbs = [0.0]
        self.widget._update_state()
        self.assertEqual(self.widget._confidence_filter.x, 1.0)
        alpha = (1 / fps) / (0.5 + 1 / fps)

        for status, brake, steer, palette in (
          (UIStatus.OVERRIDE, 0.2, 0.4, [(255, 255, 255), (82, 82, 82)]),
          (UIStatus.ENGAGED, 0.05, 0.1, [(0, 255, 204), (0, 255, 38)]),
        ):
          self.set_mads_status(status)
          predictions.brakeDisengageProbs = [brake]
          predictions.steerOverrideProbs = [steer]
          start = self.widget._confidence_filter.x
          target = (1 - predictions.brakeDisengageProbs[0]) * (1 - predictions.steerOverrideProbs[0])
          with patch.object(rl, 'draw_rectangle_rec'), patch.object(confidence, 'draw_circle_gradient') as dot:
            for frame in range(1, fps + 1):
              self.widget._update_state()
              expected = target + (start - target) * (1 - alpha) ** frame
              self.assertAlmostEqual(self.widget._confidence_filter.x, expected, places=6)
              self.widget._render(rl.Rectangle(1840, 30, 120, 960))
              self.assertEqual([(c.r, c.g, c.b) for c in dot.call_args.args[-2:]], palette)
          self.assertNotAlmostEqual(self.widget._confidence_filter.x, target, places=3)

  def test_every_mads_status_requires_fresh_sp_message_and_recovers_from_current_data(self):
    for status in UIStatus:
      for attr, invalid in (('valid', False), ('alive', False), ('recv_frame', 10),
                            ('recv_time', 9.49), ('recv_time', 11.0)):
        with self.subTest(status=status, attr=attr):
          self.widget = confidence.ModelConfidenceRenderer()
          self.set_mads_status(status)
          predictions = self.sm['modelV2'].meta.disengagePredictions
          predictions.brakeDisengageProbs = [0.0]
          predictions.steerOverrideProbs = [0.0]
          self.widget._update_state()
          self.assertTrue(self.widget._has_value)
          self.assertEqual(self.widget._confidence_filter.x, -0.5 if status == UIStatus.DISENGAGED else 1.0)

          field = getattr(self.sm, attr)
          previous = field['selfdriveStateSP']
          field['selfdriveStateSP'] = invalid
          self.widget._update_state()
          self.assertFalse(self.widget._has_value)
          self.assertFalse(self.widget._confidence_filter.initialized)
          self.assertEqual(self.widget._confidence_filter.x, 0.0)
          rect = rl.Rectangle(1840, 30, 120, 960)
          with patch.object(rl, 'draw_rectangle_rec'), patch.object(confidence, 'draw_circle_gradient') as dot:
            self.widget._render(rect)
            _, center_y, radius = dot.call_args.args[:3]
            self.assertEqual(center_y + math.ceil(radius * math.sqrt(2)) + 1, rect.y + rect.height)
            self.assertEqual([(c.r, c.g, c.b) for c in dot.call_args.args[-2:]], [(50, 50, 50), (13, 13, 13)])

          field['selfdriveStateSP'] = previous
          predictions.brakeDisengageProbs = [0.8]
          predictions.steerOverrideProbs = [0.8]
          self.set_mads_status(status)
          self.widget._update_state()
          self.assertTrue(self.widget._has_value)
          expected = -0.5 if status == UIStatus.DISENGAGED else (
            0.2 if status in (UIStatus.LAT_ONLY, UIStatus.LONG_ONLY) else 0.04)
          self.assertAlmostEqual(self.widget._confidence_filter.x, expected, places=6)

  def test_changes_between_active_axis_domains_reset_the_value(self):
    domains = {UIStatus.ENGAGED: 'both', UIStatus.OVERRIDE: 'both',
               UIStatus.LAT_ONLY: 'lateral', UIStatus.LONG_ONLY: 'longitudinal'}
    predictions = self.sm['modelV2'].meta.disengagePredictions
    for before, before_domain in domains.items():
      for after, after_domain in domains.items():
        if before_domain == after_domain:
          continue
        with self.subTest(before=before, after=after):
          self.widget = confidence.ModelConfidenceRenderer()
          predictions.brakeDisengageProbs = [0.1]
          predictions.steerOverrideProbs = [0.2]
          self.set_mads_status(before)
          self.widget._update_state()
          predictions.brakeDisengageProbs = [0.6]
          predictions.steerOverrideProbs = [0.4]
          self.set_mads_status(after)
          self.widget._update_state()
          expected = {'both': 0.24, 'lateral': 0.6, 'longitudinal': 0.4}[after_domain]
          self.assertAlmostEqual(self.widget._confidence_filter.x, expected, places=6)

  def test_disengaged_entrance_and_exit_remain_continuous_between_axis_domains(self):
    predictions = self.sm['modelV2'].meta.disengagePredictions
    predictions.brakeDisengageProbs = [0.6]
    predictions.steerOverrideProbs = [0.2]
    self.set_mads_status(UIStatus.LAT_ONLY)
    self.widget._update_state()
    start = self.widget._confidence_filter.x
    alpha = self.widget._confidence_filter.alpha
    self.set_mads_status(UIStatus.DISENGAGED)
    self.widget._update_state()
    descending = (1 - alpha) * start + alpha * -0.5
    self.assertAlmostEqual(self.widget._confidence_filter.x, descending, places=6)
    self.set_mads_status(UIStatus.LONG_ONLY)
    self.widget._update_state()
    self.assertAlmostEqual(self.widget._confidence_filter.x, (1 - alpha) * descending + alpha * 0.4, places=6)
    self.assertNotAlmostEqual(self.widget._confidence_filter.x, 0.4, places=3)

  def test_alerts_do_not_hide_or_reset_confidence(self):
    self.widget._update_state()
    initial = self.widget._confidence_filter.x
    for size in (log.SelfdriveState.AlertSize.small, log.SelfdriveState.AlertSize.mid, log.SelfdriveState.AlertSize.full):
      self.sm['selfdriveState'].alertSize = size
      self.widget._update_state()
      self.assertTrue(self.widget._has_value)
      self.assertAlmostEqual(self.widget._confidence_filter.x, initial)
      with patch.object(rl, 'draw_rectangle_rec'), patch.object(confidence, 'draw_circle_gradient') as dot:
        self.widget._render(rl.Rectangle(1840, 30, 120, 960))
        dot.assert_called_once()

  def test_missing_data_is_neutral_at_bottom_not_stale_high(self):
    self.widget._update_state()
    self.sm.valid['modelV2'] = False
    self.widget._update_state()
    rect = rl.Rectangle(1840, 30, 120, 960)
    with patch.object(rl, 'draw_rectangle_rec'), patch.object(confidence, 'draw_circle_gradient') as dot:
      self.widget._render(rect)
      _, center_y, radius = dot.call_args.args[:3]
      self.assertEqual(center_y + math.ceil(radius * math.sqrt(2)) + 1, rect.y + rect.height)
      for color in dot.call_args.args[-2:]:
        self.assertEqual((color.r, color.g, color.b), (color.r,) * 3)

  def test_circle_stays_in_translucent_panel(self):
    self.widget._update_state()
    rect = rl.Rectangle(1840, 30, 120, 960)
    for value in (0.0, 0.5, 1.0):
      self.widget._confidence_filter.x = value
      with patch.object(rl, 'draw_rectangle_rec') as panel, patch.object(confidence, 'draw_circle_gradient') as circle:
        self.widget._render(rect)
        center_x, center_y, radius = circle.call_args.args[:3]
        self.assertGreaterEqual(center_x - radius, rect.x)
        self.assertLessEqual(center_x + radius, rect.x + rect.width)
        self.assertGreaterEqual(center_y - radius, rect.y)
        self.assertLessEqual(center_y + radius, rect.y + rect.height)
        color = panel.call_args.args[1]
        self.assertEqual((color.r, color.g, color.b, color.a), (0, 0, 0, 100))


if __name__ == '__main__':
  unittest.main()
