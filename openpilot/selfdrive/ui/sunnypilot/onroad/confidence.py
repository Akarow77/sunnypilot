"""Comma four style model confidence indicator for the 3X driving screen."""
import math
import time

import pyray as rl

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import Widget

CONFIDENCE_PANEL_WIDTH = 120
MODEL_MAX_AGE = 0.5
_CIRCLE_POINTS = tuple((math.cos(i * math.tau / 48), math.sin(i * math.tau / 48)) for i in range(49))


def draw_circle_gradient(center_x: float, center_y: float, radius: int, top: rl.Color, bottom: rl.Color) -> None:
  """Draw only the circle, preserving the translucent background around it."""
  rl.rl_check_render_batch_limit(48 * 3)
  rl.rl_set_texture(rl.rl_get_texture_id_default())
  rl.rl_begin(rl.RL_TRIANGLES)
  rl.rl_tex_coord2f(0.0, 0.0)
  for current, following in zip(_CIRCLE_POINTS, _CIRCLE_POINTS[1:]):
    for dx, dy in ((0.0, 0.0), following, current):
      mix = (dy + 1.0) / 2.0
      rl.rl_color4ub(*(round(a + (b - a) * mix) for a, b in
                       ((top.r, bottom.r), (top.g, bottom.g), (top.b, bottom.b), (top.a, bottom.a))))
      rl.rl_vertex2f(center_x + dx * radius, center_y + dy * radius)
  rl.rl_end()
  rl.rl_set_texture(0)


def model_confidence(status: UIStatus, brake_probs, steer_probs) -> float | None:
  """Use the active axes' intervention predictions, not a probability of safe driving."""
  arrays = (steer_probs,) if status == UIStatus.LAT_ONLY else (brake_probs,) if status == UIStatus.LONG_ONLY else (brake_probs, steer_probs)
  confidence = 1.0
  for probs in arrays:
    if not len(probs) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probs):
      return None
    confidence *= 1.0 - max(probs)
  return confidence


def confidence_colors(value: float, override: bool = False, inactive: bool = False) -> tuple[rl.Color, rl.Color]:
  if inactive:
    return rl.Color(50, 50, 50, 255), rl.Color(13, 13, 13, 255)
  if override:
    return rl.Color(255, 255, 255, 255), rl.Color(82, 82, 82, 255)
  if value > 0.5:
    return rl.Color(0, 255, 204, 255), rl.Color(0, 255, 38, 255)
  if value > 0.2:
    return rl.Color(255, 200, 0, 255), rl.Color(255, 115, 0, 255)
  return rl.Color(255, 0, 21, 255), rl.Color(255, 0, 89, 255)


class ModelConfidenceRenderer(Widget):
  def __init__(self):
    super().__init__()
    self._confidence_filter = FirstOrderFilter(0.0, 0.5, 1 / gui_app.target_fps, initialized=False)
    self._has_value = False
    self._last_status = None
    self._started_frame = None
    self.set_enabled(False)  # Display only; touches still belong to the road view.

  def _reset(self):
    self._has_value = False
    self._confidence_filter.x = 0.0
    self._confidence_filter.initialized = False

  def _update_state(self):
    sm = ui_state.sm
    status_changed = self._last_status != ui_state.status
    # Keep the position across disengage/re-engage so the original four's
    # descent and return animation are not replaced by an immediate jump.
    disengage_transition = UIStatus.DISENGAGED in (self._last_status, ui_state.status)
    if self._started_frame != ui_state.started_frame or (status_changed and not disengage_transition):
      self._reset()
    self._started_frame = ui_state.started_frame
    self._last_status = ui_state.status

    # A frozen or malformed prediction must never leave a high confidence dot onscreen.
    if not ui_state.model_confidence or not ui_state.started:
      self._reset()
      return
    now = time.monotonic()
    service = 'selfdriveState'
    if (not sm.valid[service] or not sm.alive[service] or sm.recv_frame[service] <= ui_state.started_frame or
        not 0 <= now - sm.recv_time[service] <= MODEL_MAX_AGE):
      self._reset()
      return
    if ui_state.status == UIStatus.DISENGAGED:
      # The four animates out with this target, independent of model confidence.
      # Rendering clamps at the bottom to keep the user's always-visible marker.
      self._confidence_filter.update(-0.5)
      self._has_value = True
      return
    service = 'modelV2'
    if (not sm.valid[service] or not sm.alive[service] or sm.recv_frame[service] <= ui_state.started_frame or
        not 0 <= now - sm.recv_time[service] <= MODEL_MAX_AGE):
      self._reset()
      return
    if ui_state.status in (UIStatus.LAT_ONLY, UIStatus.LONG_ONLY):
      if (not sm.valid['selfdriveStateSP'] or not sm.alive['selfdriveStateSP'] or
          sm.recv_frame['selfdriveStateSP'] <= ui_state.started_frame or
          not 0 <= now - sm.recv_time['selfdriveStateSP'] <= MODEL_MAX_AGE):
        self._reset()
        return
    predictions = sm['modelV2'].meta.disengagePredictions
    value = model_confidence(ui_state.status, predictions.brakeDisengageProbs, predictions.steerOverrideProbs)
    if value is None:
      self._reset()
      return
    self._confidence_filter.update(value)
    self._has_value = True

  def _render(self, rect: rl.Rectangle):
    if not ui_state.model_confidence or not ui_state.started or rect.width <= 0 or rect.height <= 0:
      return

    # Match the existing bottom developer bar's translucent black tint.
    rl.draw_rectangle_rec(rect, rl.Color(0, 0, 0, 100))
    radius = min(40, math.floor(min(rect.width, rect.height) / (2 * math.sqrt(2))) - 1)
    if radius < 1:
      return
    # Missing data uses a neutral marker at the bottom, never an old high value.
    value = min(1.0, max(0.0, self._confidence_filter.x)) if self._has_value else 0.0
    padding = math.ceil(radius * math.sqrt(2)) + 1
    x = int(rect.x + rect.width / 2)
    y = int(rect.y + padding + (1 - value) * (rect.height - 2 * padding))
    top, bottom = confidence_colors(value, ui_state.status == UIStatus.OVERRIDE,
                                    ui_state.status == UIStatus.DISENGAGED or not self._has_value)
    draw_circle_gradient(x, y, radius, top, bottom)
