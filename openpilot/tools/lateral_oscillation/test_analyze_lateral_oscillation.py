import unittest

import numpy as np

from openpilot.tools.lateral_oscillation.analyze_lateral_oscillation import (
  ACCELERATION_DUE_TO_GRAVITY,
  ExtractedLog,
  Signal,
  band_component,
  best_lag,
  contiguous_regions,
  resample,
  select_non_overlapping_windows,
  spectral_metrics,
)


class TestLateralOscillationAnalysis(unittest.TestCase):
  def test_detects_fast_oscillation(self):
    fs = 100.0
    t = np.arange(0.0, 10.0, 1.0 / fs)
    x = 0.2 * np.sin(2.0 * np.pi * 1.8 * t) + 0.01 * t
    metrics = spectral_metrics(x, fs)
    self.assertAlmostEqual(metrics["dominantHz"], 1.8, delta=0.11)
    self.assertAlmostEqual(metrics["dominantAmplitude"], 0.2, delta=0.02)
    self.assertGreater(metrics["bandPowerRatio"], 0.95)

  def test_spectral_metrics_tolerates_short_edge_gap(self):
    fs = 100.0
    t = np.arange(0.0, 8.0, 1.0 / fs)
    x = np.sin(2.0 * np.pi * 2.0 * t)
    x[-20:] = np.nan
    metrics = spectral_metrics(x, fs)
    self.assertAlmostEqual(metrics["dominantHz"], 2.0, delta=0.13)

  def test_response_lag_sign(self):
    fs = 100.0
    t = np.arange(0.0, 12.0, 1.0 / fs)
    reference = np.sin(2.0 * np.pi * 1.2 * t) + 0.3 * np.sin(2.0 * np.pi * 2.1 * t)
    samples = 23
    response = np.r_[np.zeros(samples), reference[:-samples]]
    delay, corr = best_lag(reference, response, fs)
    self.assertAlmostEqual(delay, samples / fs, delta=1.0 / fs)
    self.assertGreater(corr, 0.98)

  def test_band_component_rejects_slow_lane_ping_pong(self):
    fs = 100.0
    t = np.arange(0.0, 20.0, 1.0 / fs)
    slow = np.sin(2.0 * np.pi * 0.2 * t)
    fast = 0.3 * np.sin(2.0 * np.pi * 1.5 * t)
    filtered = band_component(slow + fast, fs)
    self.assertAlmostEqual(float(np.sqrt(np.mean(filtered ** 2))), 0.3 / np.sqrt(2), delta=0.02)

  def test_contiguous_regions(self):
    mask = np.array([False, True, True, False, True, True, True, False])
    self.assertEqual(contiguous_regions(mask, 3), [(4, 7)])

  def test_selects_non_overlapping_windows(self):
    windows = [
      {"_startIndex": 0, "_endIndex": 10, "_score": 3.0},
      {"_startIndex": 5, "_endIndex": 15, "_score": 2.0},
      {"_startIndex": 20, "_endIndex": 30, "_score": 1.0},
    ]
    selected = select_non_overlapping_windows(windows, 3)
    self.assertEqual([(w["_startIndex"], w["_endIndex"]) for w in selected], [(0, 10), (20, 30)])

  def test_signal_ignores_non_numeric_values(self):
    signal = Signal()
    signal.append(1.0, "not-a-number")
    signal.append(2.0, 3.0)
    self.assertEqual(signal.times, [2.0])
    self.assertEqual(signal.values, [3.0])

  def test_estimates_jerk_aware_friction_from_feedforward(self):
    extracted = ExtractedLog()
    for i in range(20):
      t = i * 0.01
      extracted.add("control_curvature", t, 0.0)
      extracted.add("desired_lat_accel", t, 0.6)
      extracted.add("pid_f", t, (0.6 - ACCELERATION_DUE_TO_GRAVITY * 0.01) / 2.0 + 0.08)
      extracted.add("lat_accel_factor", t, 2.0)
      extracted.add("lat_accel_offset", t, 0.0)
      extracted.add("friction", t, 0.1)
      extracted.add("actual_lat_accel", t, 0.3)
      extracted.add("cmd_torque", t, -0.5)
      extracted.add("roll", t, 0.01)
    data = resample(extracted)
    np.testing.assert_allclose(data["estimated_jerk_friction_torque"], 0.08, atol=1e-12)
    np.testing.assert_allclose(data["estimated_patched_jerk_friction_torque"], 0.1, atol=1e-12)
    np.testing.assert_allclose(data["counterfactual_patched_jerk_cmd_torque"], -0.52, atol=1e-12)
    np.testing.assert_allclose(data["estimated_base_v0_friction_torque"], 0.1, atol=1e-12)
    np.testing.assert_allclose(data["counterfactual_base_v0_cmd_torque"], -0.52, atol=1e-12)

  def test_builds_imu_lateral_acceleration_for_vehicle_model_check(self):
    extracted = ExtractedLog()
    for i in range(20):
      t = i * 0.01
      extracted.add("control_curvature", t, 0.0)
      extracted.add("v_ego", t, 25.0)
      extracted.add("yaw_rate", t, 0.02)
      extracted.add("device_roll", t, 0.01)
      extracted.add("roll", t, 0.01)
      extracted.add("actual_lat_accel", t, 0.45)
    data = resample(extracted)
    np.testing.assert_allclose(data["inertial_lat_accel"], 0.5, atol=1e-12)
    expected_gravity_adjusted = 0.5 - ACCELERATION_DUE_TO_GRAVITY * np.sin(0.01)
    np.testing.assert_allclose(data["gravity_adjusted_inertial_lat_accel"], expected_gravity_adjusted, atol=1e-12)


if __name__ == "__main__":
  unittest.main()
