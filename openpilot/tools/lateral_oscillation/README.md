# Lateral oscillation offline diagnostic

This is a read-only diagnostic for already-recorded openpilot `rlog.zst` or
`qlog.zst` files. It does not connect to a car or publish actuator commands.
An `rlog` is strongly preferred because qlogs can omit or decimate the fast
control signals needed to distinguish steering vibration from slow ping-pong.

Run from the sunnypilot repository root:

```sh
PYTHONPATH=. .venv/bin/python openpilot/tools/lateral_oscillation/analyze_lateral_oscillation.py \
  /path/to/route-or-rlog --output-dir /tmp/niro-lateral-report
```

Outputs:

- `report.html`: runtime settings plus plots for the strongest 0.5–5 Hz windows.
- `report.json`: machine-readable frequency, amplitude, delay/correlation,
  saturation, and effective torque/delay parameter metrics.
- `top_windows.csv`: aligned raw signals for independent plotting.

The default filter requires at least 80 km/h, lateral control active, and no
driver steering override. The tool reports evidence; it does not recommend PID,
friction, delay, or EPS-limit changes. Any tuning decision must be based on the
actual Niro Plus route and offline replay, never an intentional high-speed test.

Interpret the signal chain in this order:

1. If `model_lat_accel` already contains the dominant oscillation and leads
   `control_lat_accel`, investigate the driving-model/path output and model timing.
2. If desired lateral acceleration is quiet but `actual_lat_accel`, PID error/P,
   steering torque, and steering rate oscillate, investigate closed-loop gain,
   effective torque parameters, delay, and vehicle dynamics.
3. A high `commandOutputMismatchFraction` means the Hyundai car-controller
   driver/rate limits materially changed the request. It does not by itself prove
   that the limiter caused the oscillation; compare phase and sign reversals.
4. When the recorded `LateralJerkTorqueController` toggle is on,
   `estimatedJerkFrictionBandPowerShare` subtracts the linear desired-acceleration
   feed-forward from the logged F term and estimates the remaining friction/jerk
   contribution. `feedforwardBandPowerShare` is the unseparated F term. Confirm
   error zero-crossings and the runtime toggle before blaming friction compensation.
   `oneStepBaseV0CommandBandRmsRatio` replaces only the logged Jerk-aware
   friction/offset terms with the standard V0 formula. It is a one-step offline
   comparison, not a closed-loop replay and not a prediction of road behavior.
   `oneStepPatchedJerkCommandBandRmsRatio` evaluates the retained 0.7
   acceleration-error factor for zero-lookahead windows using the same limitation.
5. Self-Tune and live delay are slow estimators. Use their recorded min/median/max
   values to find a wrong effective calibration or a mode mismatch, not as proof
   that they directly generated a 0.5–5 Hz signal.
6. Steering-angle/driver-torque vibration with a quiet openpilot torque request,
   or vibration that also exists with steering assistance disengaged in ordinary
   driving, points outside the controller (EPS, tires/wheels, alignment, or other
   mechanical causes).
7. `vehicleModelInertialBandGain` compares the controller's steering-angle-based
   lateral acceleration with calibrated IMU yaw-rate lateral acceleration. A
   repeatable gain far from 1.0 with high correlation points to vehicle-model
   parameters (steer ratio/tire stiffness) or steering-angle scaling. Low
   correlation is not evidence for changing those parameters; inspect timing,
   road input, and sensor validity first.
