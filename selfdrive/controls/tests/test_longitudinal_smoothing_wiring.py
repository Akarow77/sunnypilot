import inspect
import re
from pathlib import Path

from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_smoothing_params_default_off():
  params_keys = (REPO_ROOT / "common/params_keys.h").read_text()

  assert re.search(r'"AccelPersonalityEnabled", \{PERSISTENT \| BACKUP, BOOL, "0"\}', params_keys)
  assert re.search(r'"RadarDistance", \{PERSISTENT \| BACKUP, BOOL, "0"\}', params_keys)
  # the retired dRel-bias sub-toggles must not return (their features were deleted in the input-shaping rewrite)
  assert '"StopGapBias"' not in params_keys
  assert '"LeadDecelAnticipate"' not in params_keys


def test_output_is_byte_stock_and_inputs_are_shaped():
  update_src = inspect.getsource(LongitudinalPlanner.update)

  # INPUT shaping only: the accel ceiling and the radar-conditioning seam are present...
  assert "self.accel.get_max_accel(v_ego)" in update_src
  assert "self.mpc.update(self.smooth_radarstate(sm['radarState'])" in update_src
  # ...and the OUTPUT is never post-shaped (byte-stock output; no accel shaping, no should_stop override).
  assert "smooth_target_accel" not in update_src
  assert "sng_should_stop" not in update_src   # reverted: the should_stop hysteresis caused a high-speed under-brake


def test_t_follow_hook_wired_and_identity_default():
  init_src = inspect.getsource(LongitudinalPlanner.__init__)
  assert "self.mpc.t_follow_fn = self.accel.get_t_follow" in init_src   # planner wires the add-only widen

  mpc_init = inspect.getsource(LongitudinalMpc.__init__)
  assert "self.t_follow_fn = None" in mpc_init                          # default None == byte-stock identity

  mpc_update = inspect.getsource(LongitudinalMpc.update)
  assert "if self.t_follow_fn is not None:" in mpc_update               # guarded hook, only fires when set


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
