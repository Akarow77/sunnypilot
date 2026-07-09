"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Acceleration Personality tuning tables. The controller shapes only what the longitudinal MPC CONSUMES
(the positive-accel ceiling + its open-rate, and an add-only follow-gap widen fed to the MPC's t_follow);
it never post-shapes the MPC's output accel. Disabled => every getter returns the upstream stock value,
so off == byte-stock.
"""

from cereal import custom

AccelerationPersonality = custom.LongitudinalPlanSP.AccelerationPersonality
ECO = AccelerationPersonality.eco
NORMAL = AccelerationPersonality.normal
SPORT = AccelerationPersonality.sport

PERSONALITY_MIN = min(AccelerationPersonality.schema.enumerants.values())
PERSONALITY_MAX = max(AccelerationPersonality.schema.enumerants.values())

# --- Positive-accel ceiling (launch/cruise) + its upward open-rate ---------------------------------------
# Disabled -> falls back to STOCK_* (byte-stock). ACCEL_MAX (opendbc) hard-caps the ceiling at 2.0 m/s^2.
A_CRUISE_MAX_BP = [0., 10., 25., 40.]              # m/s (matches upstream A_CRUISE_MAX_BP)
STOCK_A_CRUISE_MAX_V = [1.6, 1.2, 0.8, 0.6]        # upstream A_CRUISE_MAX_VALS
STOCK_RISE_RATE = 0.05                             # upstream ceiling open-rate (m/s^2 per cycle)
A_CRUISE_MAX_V = {
  ECO:    [1.55, 0.75, 0.35, 0.20],
  NORMAL: [2.00, 1.40, 0.95, 0.70],
  SPORT:  [2.00, 1.70, 1.20, 0.90],
}
# Ceiling open-rate (m/s^2 per cycle): fast near v=0 so launch isn't delayed, tapers by v=5.
RISE_RATE_BP = [0., 5.]                            # m/s
RISE_RATE_V = {
  ECO:    [0.80, 0.07],
  NORMAL: [1.00, 0.16],
  SPORT:  [1.20, 0.24],
}
# The fast near-stop rise-rate above only looks at v_ego, so it also fires while still closing on a lead
# that hasn't cleared yet, letting the ceiling snap open faster than the actual situation calls for. Fall
# back to the stock rate whenever still closing (vRel this negative or more).
RISE_RATE_LEAD_VREL_GATE = -0.5                    # m/s

# --- Launch jerk-cost relaxation (MPC INPUT: scales the core MPC's own jerk_factor) -----------------------
# Bounded near a stop, ramped back to 1.0 (stock) by cruise speed. v=0 knot closed-loop verified; SPORT tied
# to NORMAL rather than pushed lower (lower destabilizes the solver at this knot).
JERK_SCALE_BP = [0., 5.]                           # m/s
JERK_SCALE_V = {
  ECO:    [0.60, 1.0],
  NORMAL: [0.45, 1.0],
  SPORT:  [0.45, 1.0],
}

# --- Onset jerk-cost relaxation (MPC INPUT: any accel<->decel direction change, not just launch) ----------
ONSET_DEADBAND = 0.15          # m/s^2: ignore aEgo noise this small around a zero-crossing
ONSET_RAMP_S = 0.4             # s: ease back to stock over this long
ONSET_FLOOR = {ECO: 0.75, NORMAL: 0.65, SPORT: 0.50}

# Ramp shared by the two level-triggered factors below (lead-braking, closing-rate): a sustained pinned
# floor destabilizes the MPC's re-solve, so both ease back to stock over this window instead.
RELAX_RAMP_S = 0.4

# --- Lead-braking jerk-cost relaxation (MPC INPUT: react faster to a hard-braking lead) --------------------
LEAD_BRAKE_ALEAD_BP = [-3.0, -0.5]      # m/s^2, lead's own aLeadK
LEAD_BRAKE_FACTOR_V = {
  ECO:    [0.75, 1.0],
  NORMAL: [0.60, 1.0],
  SPORT:  [0.45, 1.0],
}

# --- Closing-rate jerk-cost relaxation (MPC INPUT: react faster to a fast-closing gap, any cause) ----------
CLOSING_VREL_BP = [-6.0, -1.5]          # m/s, closing rate (negative = closing)
CLOSING_FACTOR_V = {
  ECO:    [0.75, 1.0],
  NORMAL: [0.60, 1.0],
  SPORT:  [0.45, 1.0],
}

# --- Follow-gap widen (add-only, fed to the MPC t_follow) ------------------------------------------------
TF_WIDEN_V_BP = [14.0, 28.0]                       # m/s: widen ramps in across this band, flat above
TF_WIDEN_BASE_V = [0.0, 0.30]                      # s: base follow-time added at the band ends (pre-tier)
TF_WIDEN_TIER = {ECO: 1.30, NORMAL: 1.00, SPORT: 0.50}
TF_WIDEN_MAX = 0.45                                # s: absolute cap on the added gap
TF_SLEW_PER_S = 0.50                               # s per second: max rate the widen may open/close
TF_DECEL_HOLD_A = -0.20                            # m/s^2: at/below this a_ego the widen won't shrink
