"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from cereal import custom

AccelerationPersonality = custom.LongitudinalPlanSP.AccelerationPersonality
ECO = AccelerationPersonality.eco
NORMAL = AccelerationPersonality.normal
SPORT = AccelerationPersonality.sport

PERSONALITY_MIN = min(AccelerationPersonality.schema.enumerants.values())
PERSONALITY_MAX = max(AccelerationPersonality.schema.enumerants.values())

# Accel ceiling. NORMAL is stock so a disabled controller (forced to NORMAL) is stock.
A_CRUISE_MAX_BP = [0., 10., 25., 40.]
STOCK_A_CRUISE_MAX_V = [1.6, 1.2, 0.8, 0.6]
STOCK_RISE_RATE = 0.05
A_CRUISE_MAX_V = {
  ECO:    [1.20, 0.85, 0.45, 0.30],
  NORMAL: STOCK_A_CRUISE_MAX_V,
  SPORT:  [1.75, 1.30, 0.90, 0.65],
}
RISE_RATE = {ECO: 0.02, NORMAL: STOCK_RISE_RATE, SPORT: 0.06}

# Early soft braking: predicted brake need (m/s^2) -> early decel target (m/s^2).
SMOOTH_DECEL_BP = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4]
SMOOTH_DECEL_V = {
  ECO:    [0.00, -0.10, -0.26, -0.52, -0.78, -1.00, -1.20],
  NORMAL: [0.00, -0.15, -0.36, -0.68, -1.00, -1.25, -1.50],
  SPORT:  [0.00, -0.17, -0.40, -0.72, -1.05, -1.35, -1.65],
}
BRAKE_DEEPENING_JERK = {ECO: 0.5, NORMAL: 0.8, SPORT: 1.0}
BRAKE_RELEASE_JERK = 2.0
ACCEL_RISE_JERK = {ECO: 0.7, NORMAL: 1.2, SPORT: 1.6}

SMOOTH_DECEL_LOOKAHEAD_T = 3.0
MIN_SMOOTH_BRAKE_NEED = 0.3
HARD_BRAKE_TARGET_ACCEL = -2.0
HARD_BRAKE_NEED = 2.6
