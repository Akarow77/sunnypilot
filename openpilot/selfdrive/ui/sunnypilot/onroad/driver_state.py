"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math
import time

import pyray as rl

from openpilot.selfdrive.ui import UI_BORDER_SIZE
from openpilot.selfdrive.ui.mici.onroad.driver_state import AlertSize, DriverStateRenderer
from openpilot.selfdrive.ui.sunnypilot.onroad.confidence import CONFIDENCE_PANEL_WIDTH
from openpilot.selfdrive.ui.sunnypilot.onroad.developer_ui import get_bottom_dev_ui_offset
from openpilot.selfdrive.ui.sunnypilot.onroad.rocket_fuel import RocketFuel
from openpilot.selfdrive.ui.ui_state import UIStatus, ui_state
from openpilot.system.ui.widgets import Widget

DM_ICON_SIZE = 192
DM_MAX_AGE = 0.5
DM_SERVICES = ("driverMonitoringState", "driverStateV2", "selfdriveState")


class DriverStateRendererSP(Widget):
  """Place the unchanged comma four DM renderer in the comma 3X HUD."""
  def __init__(self):
    super().__init__()
    self._driver_state_renderer = DriverStateRenderer()
    self._data_valid = False
    self._started_frame = None
    self.set_enabled(False)

  def _reset(self):
    renderer = self._driver_state_renderer
    renderer.set_should_draw(False)
    renderer._is_active = False
    renderer._face_detected = False
    renderer._looking_center = False
    renderer._awareness_unfull = False
    renderer._fade_filter.x = 0.0
    renderer._color_fade_filter.x = 1.0
    renderer._looking_center_filter.x = 0.0
    # Do not blend a new drive or recovered stream with the previous pose.
    for f in (renderer._pitch_filter, renderer._yaw_filter, renderer._rotation_filter):
      f.x = 0.0
      f.initialized = False

  def _update_state(self):
    sm = ui_state.sm
    if self._started_frame != ui_state.started_frame:
      self._reset()
      self._started_frame = ui_state.started_frame
    now = time.monotonic()
    self._data_valid = ui_state.started and all(
      sm.valid[service] and sm.alive[service] and sm.recv_frame[service] > ui_state.started_frame and
      0 <= now - sm.recv_time[service] <= DM_MAX_AGE
      for service in DM_SERVICES
    )
    if self._data_valid:
      vision = sm["driverMonitoringState"].visionPolicyState
      self._data_valid = all(math.isfinite(value) for value in
                             (vision.pose.pitch, vision.pose.yaw, vision.awarenessPercent)) and 0 <= vision.awarenessPercent <= 100
    if not self._data_valid:
      self._reset()
      return

    renderer = self._driver_state_renderer
    renderer.set_should_draw(ui_state.status != UIStatus.DISENGAGED or ui_state.always_on_dm)
    # Reuse four's pose calibration, RHD sign, rotation smoothing and awareness
    # transitions; this adapter only handles the 3X layout and input validity.
    renderer._update_state()

  def _render(self, rect: rl.Rectangle):
    if not self._data_valid:
      return

    renderer = self._driver_state_renderer
    # Stock 3X hides DM for every alert size, including lane-change notices.
    # DM is drawn after alerts here, so suppress the fade immediately as well.
    if ui_state.sm["selfdriveState"].alertSize != AlertSize.none:
      renderer._fade_filter.x = 0.0
      return

    right_inset = CONFIDENCE_PANEL_WIDTH if ui_state.model_confidence else 0
    width = max(0.0, rect.width - right_inset)
    height = max(0.0, rect.height - get_bottom_dev_ui_offset())
    size = min(DM_ICON_SIZE, max(0.0, min(width, height) - 2 * UI_BORDER_SIZE))
    if size <= 0:
      return

    left_margin = max(UI_BORDER_SIZE, RocketFuel.BAR_WIDTH + 16) if ui_state.rocket_fuel else UI_BORDER_SIZE
    x = rect.x + (width - UI_BORDER_SIZE - size if renderer.is_rhd else left_margin)
    y = rect.y + height - UI_BORDER_SIZE - size
    resized = renderer.rect.width != size or renderer.rect.height != size
    renderer.set_rect(rl.Rectangle(x, y, size, size))
    if resized:
      renderer.load_icons()

    renderer._render(renderer.rect)
