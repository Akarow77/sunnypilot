"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

LongitudinalPlannerSP.is_e2e() decides whether the e2e model's raw action.desiredAcceleration blends into
the MPC's solution via min(). A near/fast-closing radar lead must always route to pure MPC regardless of
whether DEC itself is on -- that baseline previously lived entirely inside DEC's active()-gated branch, so
turning DEC off silently dropped it (identical lead input, different is_e2e() answer). These tests pin the
fix: the lead check now runs unconditionally, before DEC's own toggle is even consulted.
"""

from types import SimpleNamespace

from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP


class FakeDec:
  def __init__(self, active=False, mode="acc", has_radar_acc_lead=False):
    self._active = active
    self._mode = mode
    self._has_radar_acc_lead = has_radar_acc_lead

  def active(self):
    return self._active

  def mode(self):
    return self._mode

  def has_radar_acc_lead(self):
    return self._has_radar_acc_lead


def make_sm(experimental_mode=True):
  return {'selfdriveState': SimpleNamespace(experimentalMode=experimental_mode)}


def is_e2e(dec):
  # is_e2e only reads self.dec -- no need to construct the full LongitudinalPlannerSP.
  return LongitudinalPlannerSP.is_e2e(SimpleNamespace(dec=dec), make_sm())


def test_experimental_mode_off_never_e2e():
  sm_off = make_sm(experimental_mode=False)
  assert not LongitudinalPlannerSP.is_e2e(SimpleNamespace(dec=FakeDec(active=True, mode="blended")), sm_off)
  assert not LongitudinalPlannerSP.is_e2e(SimpleNamespace(dec=FakeDec(has_radar_acc_lead=True)), sm_off)


def test_lead_present_blocks_e2e_regardless_of_dec_active():
  # the bug this fixes: identical lead, DEC on vs off must agree.
  assert not is_e2e(FakeDec(active=True, mode="acc", has_radar_acc_lead=True))
  assert not is_e2e(FakeDec(active=False, mode="acc", has_radar_acc_lead=True))


def test_no_lead_dec_off_falls_back_to_experimental_mode():
  assert is_e2e(FakeDec(active=False, has_radar_acc_lead=False))


def test_no_lead_dec_on_follows_dec_mode():
  assert is_e2e(FakeDec(active=True, mode="blended", has_radar_acc_lead=False))
  assert not is_e2e(FakeDec(active=True, mode="acc", has_radar_acc_lead=False))
