"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

_E2ETransitionGuard bounds how fast output_a_target may drop right after DEC switches into blended mode for
a routine reason, so the e2e model's own (previously-hidden) desiredAcceleration can't produce a same-cycle
discontinuous brake. It must never limit a rise, never limit anything while smoothing is inactive, and never
apply when the caller signals an urgent/immediate transition.
"""

import pytest

from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import _E2ETransitionGuard, TRANSITION_MAX_DROP_PER_CYCLE


def test_inactive_is_identity():
  g = _E2ETransitionGuard()
  g.apply(-0.3, False)
  assert g.apply(-1.1, False) == pytest.approx(-1.1)   # smoothing inactive -> raw passthrough always


def test_first_call_seeds_without_limiting():
  g = _E2ETransitionGuard()
  assert g.apply(-1.1, True) == pytest.approx(-1.1)    # no prior baseline -> nothing to limit against yet


def test_limits_downward_jump_when_active():
  g = _E2ETransitionGuard()
  g.apply(-0.3, False)                                 # establish baseline while inactive
  out = g.apply(-1.1, True)                             # a farther-in-one-cycle jump, smoothing now active
  assert out == pytest.approx(-0.3 - TRANSITION_MAX_DROP_PER_CYCLE)
  assert out > -1.1                                     # not the raw discontinuous value


def test_never_limits_a_rise():
  g = _E2ETransitionGuard()
  g.apply(-1.0, False)
  out = g.apply(0.5, True)                              # accel rising -- must never be held back
  assert out == pytest.approx(0.5)


def test_converges_to_raw_within_a_few_cycles():
  g = _E2ETransitionGuard()
  g.apply(-0.3, False)
  out = -0.3
  for _ in range(20):
    out = g.apply(-1.1, True)
  assert out == pytest.approx(-1.1, abs=1e-6)            # eventually tracks the sustained raw value exactly


def test_replays_real_route_e2e_transition():
  # route 550a71ee4c7a7fbe/0000049f--71203acd12, t~165.3-166.2: DEC switches acc->blended (routine slow-down,
  # not FCW) and the raw blended output snaps -0.312 -> -1.109 in one 50ms tick, then continues to ~-1.33.
  g = _E2ETransitionGuard()
  raw_acc = [-0.312, -0.312]           # still acc mode
  raw_blended = [-1.109, -1.112, -1.121, -1.110, -1.118, -1.137, -1.140, -1.133, -1.145, -1.165,
                 -1.199, -1.227, -1.255, -1.268, -1.287, -1.293, -1.326]
  for v in raw_acc:
    g.apply(v, False)
  guarded = [g.apply(v, i < 10) for i, v in enumerate(raw_blended)]   # smoothing active for the first 10 frames
  assert guarded[0] == pytest.approx(-0.312 - TRANSITION_MAX_DROP_PER_CYCLE)   # graded, not the -1.109 snap
  assert min(guarded[:3]) > -1.0        # nowhere near the raw value in the first few frames
  assert guarded[-1] == pytest.approx(raw_blended[-1])   # long since converged and tracking raw exactly


def test_reset_drops_stale_baseline():
  g = _E2ETransitionGuard()
  g.apply(-0.3, False)
  g.reset()
  assert g.apply(-1.1, True) == pytest.approx(-1.1)      # no stale baseline to limit against after reset
