import inspect
import re
from pathlib import Path

from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_smoothing_params_default_off():
  params_keys = (REPO_ROOT / "common/params_keys.h").read_text()

  assert re.search(r'"AccelPersonalityEnabled", \{PERSISTENT \| BACKUP, BOOL, "0"\}', params_keys)
  assert re.search(r'"RadarDistance", \{PERSISTENT \| BACKUP, BOOL, "0"\}', params_keys)
  assert re.search(r'"StopGapBias", \{PERSISTENT \| BACKUP, BOOL, "0"\}', params_keys)


def test_longitudinal_smoothing_stays_planner_side():
  update_src = inspect.getsource(LongitudinalPlanner.update)

  accel_ceiling_idx = update_src.index("self.accel.get_max_accel(v_ego)")
  radar_smoothing_idx = update_src.index("self.mpc.update(self.smooth_radarstate(sm['radarState'])")
  accel_smoothing_idx = update_src.index("self.accel.smooth_target_accel(")

  assert accel_ceiling_idx < radar_smoothing_idx
  assert radar_smoothing_idx < accel_smoothing_idx


# Tokens for the reverted input-side DEC model-stop-target (capped v_target into the MPC pre-solve). It was
# superseded by DEC blended-mode and chased a source-fixed radar gate; it must not silently return.
_DEC_MODEL_STOP_TOKENS = ("apply_model_stop_target", "force_stop_requested", "_update_model_stop", "MODEL_STOP_TARGET_TIME")


def test_dec_model_stop_target_not_reintroduced():
  this_file = Path(__file__).resolve()
  for sub in ("selfdrive/controls", "sunnypilot/selfdrive/controls"):
    for path in (REPO_ROOT / sub).rglob("*.py"):
      if path.resolve() == this_file:
        continue                                      # this guard names the tokens as strings
      src = path.read_text()
      for token in _DEC_MODEL_STOP_TOKENS:
        assert token not in src, f"reverted DEC model-stop-target ({token}) re-introduced in {path}"


def test_long_feature_gates():
  # The surviving opt-in long features default OFF (byte-stock until enabled + on-road verified):
  # AccelController jerk-limiter and RadarDistance lead-smoother are module flags; the StopGapBias stop-gap
  # rides on a param (its default is checked in test_smoothing_params_default_off).
  from openpilot.sunnypilot.selfdrive.controls.lib.accel_personality.constants import JERK_LIMIT_ENABLED
  from openpilot.sunnypilot.selfdrive.controls.lib.radar_distance.radar_distance import LEAD_SMOOTH_ENABLED

  assert JERK_LIMIT_ENABLED is False
  assert LEAD_SMOOTH_ENABLED is False
