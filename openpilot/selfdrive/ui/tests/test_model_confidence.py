import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pyray as rl

from openpilot.cereal import messaging, log
from openpilot.selfdrive.ui.ui_state import UIStatus
from openpilot.selfdrive.ui.sunnypilot.onroad import confidence


class TestConfidenceValue(unittest.TestCase):
  def test_active_axes(self):
    self.assertAlmostEqual(confidence.model_confidence(UIStatus.ENGAGED, [0.2, 0.1], [0.3, 0.1]), 0.56)
    self.assertAlmostEqual(confidence.model_confidence(UIStatus.LAT_ONLY, [], [0.3]), 0.7)
    self.assertAlmostEqual(confidence.model_confidence(UIStatus.LONG_ONLY, [0.2], []), 0.8)
    self.assertIsNone(confidence.model_confidence(UIStatus.DISENGAGED, [0], [0]))

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

  def test_lateral_only_color_changes_with_value(self):
    for prediction, expected in ((0.1, (0, 255, 204)), (0.7, (255, 200, 0)), (0.9, (255, 0, 21))):
      self.widget._reset()
      self.sm['modelV2'].meta.disengagePredictions.steerOverrideProbs = [prediction] * 5
      self.widget._update_state()
      self.assertTrue(self.widget._has_value)
      color = confidence.confidence_colors(self.widget._confidence_filter.x)[0]
      self.assertEqual((color.r, color.g, color.b), expected)

  def test_stale_value_hidden_and_not_reused_on_recovery(self):
    self.widget._update_state()
    self.assertGreater(self.widget._confidence_filter.x, 0.8)
    self.sm.recv_time['modelV2'] = 9.49
    self.widget._update_state()
    self.assertFalse(self.widget._has_value)
    self.sm.recv_time['modelV2'] = 10.0
    self.sm['modelV2'].meta.disengagePredictions.steerOverrideProbs = [0.95]
    self.widget._update_state()
    self.assertAlmostEqual(self.widget._confidence_filter.x, 0.05, places=6)

  def test_invalid_dead_old_session_and_future_data_hidden(self):
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

  def test_alert_offroad_disabled_and_disengaged_hidden(self):
    for attr, value in (('started', False), ('model_confidence', False), ('status', UIStatus.DISENGAGED)):
      with self.subTest(attr=attr):
        previous = getattr(self.state, attr)
        setattr(self.state, attr, value)
        self.widget._update_state()
        self.assertFalse(self.widget._has_value)
        setattr(self.state, attr, previous)
    for size in (log.SelfdriveState.AlertSize.small, log.SelfdriveState.AlertSize.mid, log.SelfdriveState.AlertSize.full):
      self.sm['selfdriveState'].alertSize = size
      self.widget._update_state()
      self.assertFalse(self.widget._has_value)

  def test_display_mask_stays_in_panel(self):
    self.widget._update_state()
    rect = rl.Rectangle(1840, 30, 120, 960)
    for value in (0.0, 0.5, 1.0):
      self.widget._confidence_filter.x = value
      with patch.object(rl, 'draw_rectangle_rec'), patch.object(rl, 'draw_rectangle_gradient_v'), patch.object(rl, 'draw_ring') as ring:
        self.widget._render(rect)
        center, _, outer = ring.call_args.args[:3]
        self.assertGreaterEqual(center.x - outer, rect.x)
        self.assertLessEqual(center.x + outer, rect.x + rect.width)
        self.assertGreaterEqual(center.y - outer, rect.y)
        self.assertLessEqual(center.y + outer, rect.y + rect.height)


if __name__ == '__main__':
  unittest.main()
