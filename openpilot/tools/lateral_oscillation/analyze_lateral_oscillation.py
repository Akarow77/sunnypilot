#!/usr/bin/env python3
"""Read-only analysis for fast lateral-control oscillations in openpilot logs.

This tool never connects to a vehicle and never publishes cereal messages. It only
reads existing rlog/qlog files and writes a report under --output-dir.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext_base import LAT_ACCEL_FRICTION_FACTOR
from openpilot.selfdrive.locationd.helpers import Pose, PoseCalibrator
from openpilot.tools.lib.logreader import LogReader


OSCILLATION_BAND_HZ = (0.5, 5.0)
ACCELERATION_DUE_TO_GRAVITY = 9.80665
V0_FRICTION_THRESHOLD = 0.3
JERK_AWARE_FRICTION_THRESHOLD = 0.2
PARAM_KEYS = (
  "EnforceTorqueControl",
  "TorqueControlTune",
  "NeuralNetworkLateralControl",
  "LateralJerkTorqueController",
  "LiveTorqueParamsToggle",
  "LiveTorqueParamsRelaxedToggle",
  "CustomTorqueParams",
  "TorqueParamsOverrideEnabled",
  "TorqueParamsOverrideLatAccelFactor",
  "TorqueParamsOverrideFriction",
  "LagdToggle",
  "LagdToggleDelay",
  "LagdValueCache",
  "ModelManager_ActiveBundle",
  "ModelManager_ActiveBundleChestnut",
  "GpuOnLanEnabled",
  "GpuOnLanActive",
  "GpuOnLanModel",
)


@dataclass
class Signal:
  times: list[float] = field(default_factory=list)
  values: list[float] = field(default_factory=list)

  def append(self, t: float, value: Any) -> None:
    try:
      value = float(value)
    except (TypeError, ValueError):
      return
    if math.isfinite(value):
      self.times.append(float(t))
      self.values.append(value)


@dataclass
class ExtractedLog:
  signals: dict[str, Signal] = field(default_factory=dict)
  car_params: dict[str, Any] = field(default_factory=dict)
  car_params_sp: dict[str, Any] = field(default_factory=dict)
  init_params: dict[str, str] = field(default_factory=dict)
  git: dict[str, Any] = field(default_factory=dict)
  message_counts: dict[str, int] = field(default_factory=dict)

  def add(self, name: str, t: float, value: Any) -> None:
    self.signals.setdefault(name, Signal()).append(t, value)


CONTINUOUS_SIGNALS = (
  "v_ego", "steering_angle_deg", "steering_rate_deg", "driver_torque", "eps_torque",
  "model_curvature", "control_curvature", "actual_curvature", "cmd_torque", "out_torque",
  "torque_output_can", "pid_error", "pid_p", "pid_i", "pid_d", "pid_f", "pid_output",
  "desired_lat_accel", "actual_lat_accel", "desired_lat_jerk", "lateral_delay",
  "lateral_delay_estimate", "lateral_delay_std", "lat_accel_factor", "lat_accel_offset",
  "friction", "steer_ratio", "stiffness_factor", "roll", "angle_offset_deg",
  "yaw_rate", "device_roll",
  "model_execution_time", "frame_drop_percent", "frame_age",
)
DISCRETE_SIGNALS = ("lat_active", "steering_pressed", "saturated", "torque_version", "torque_use_params", "delay_status",
                    "model_big", "camera_frame_dropped")


def _enum_number(value: Any) -> float:
  raw = getattr(value, "raw", value)
  try:
    return float(raw)
  except (TypeError, ValueError):
    return math.nan


def _decode_param(value: bytes) -> str:
  try:
    return value.decode("utf-8")
  except (UnicodeDecodeError, AttributeError):
    return "<binary>"


def _model_fields(raw: str | None) -> dict[str, str]:
  if not raw or raw == "<binary>":
    return {}
  try:
    bundle = json.loads(raw)
  except json.JSONDecodeError:
    return {}
  result = {key: str(bundle[key]) for key in ("displayName", "internalName", "ref", "runner", "generation") if bundle.get(key) is not None}
  models = bundle.get("models", [])
  if "displayName" not in result:
    for model in models:
      if isinstance(model, dict) and model.get("name"):
        result["displayName"] = str(model["name"])
        break
  return result


def extract_logs(paths: list[Path]) -> ExtractedLog:
  out = ExtractedLog()
  pose_calibrator = PoseCalibrator()
  params_seen = False
  car_params_seen = False
  car_params_sp_seen = False

  for event in LogReader([str(p) for p in paths], sort_by_time=True, only_union_types=True):
    which = event.which()
    out.message_counts[which] = out.message_counts.get(which, 0) + 1
    t = event.logMonoTime * 1e-9
    msg = getattr(event, which)

    if which == "initData" and not params_seen:
      out.git = {
        "version": msg.version,
        "commit": msg.gitCommit,
        "branch": msg.gitBranch,
        "remote": msg.gitRemote,
        "dirty": bool(msg.dirty),
      }
      entries = {entry.key: bytes(entry.value) for entry in msg.params.entries}
      model_keys = ("ModelManager_ActiveBundle", "ModelManager_ActiveBundleChestnut")
      out.init_params = {key: _decode_param(entries[key]) for key in PARAM_KEYS if key in entries and key not in model_keys}
      for key in model_keys:
        if key in entries:
          for field_name, value in _model_fields(_decode_param(entries[key])).items():
            out.init_params[f"{key}.{field_name}"] = value
      params_seen = True

    elif which == "carParams" and not car_params_seen:
      torque = msg.lateralTuning.torque if msg.lateralTuning.which() == "torque" else None
      out.car_params = {
        "carFingerprint": msg.carFingerprint,
        "brand": msg.brand,
        "steerControlType": str(msg.steerControlType),
        "lateralTuning": msg.lateralTuning.which(),
        "steerActuatorDelay": float(msg.steerActuatorDelay),
        "steerLimitTimer": float(msg.steerLimitTimer),
        "steerRatio": float(msg.steerRatio),
        "tireStiffnessFactor": float(msg.tireStiffnessFactor),
        "mass": float(msg.mass),
        "wheelbase": float(msg.wheelbase),
        "offlineLatAccelFactor": float(torque.latAccelFactor) if torque is not None else None,
        "offlineFriction": float(torque.friction) if torque is not None else None,
        "offlineSteeringAngleDeadzoneDeg": float(torque.steeringAngleDeadzoneDeg) if torque is not None else None,
      }
      car_params_seen = True

    elif which == "carParamsSP" and not car_params_sp_seen:
      nn_model = msg.neuralNetworkLateralControl.model
      out.car_params_sp = {
        "nnlcModelName": nn_model.name,
        "nnlcModelPath": nn_model.path,
        "nnlcFuzzyFingerprint": bool(msg.neuralNetworkLateralControl.fuzzyFingerprint),
      }
      car_params_sp_seen = True

    elif which == "modelV2":
      out.add("model_curvature", t, msg.action.desiredCurvature)
      out.add("model_execution_time", t, msg.modelExecutionTime)
      out.add("frame_drop_percent", t, msg.frameDropPerc)
      out.add("frame_age", t, msg.frameAge)
      out.add("model_big", t, msg.big)

    elif which == "cameraOdometry":
      out.add("camera_frame_dropped", t, getattr(msg, "frameDropped", False))

    elif which == "extrinsicsCalibration":
      pose_calibrator.feed_extrinsics_calibration(msg)

    elif which == "deviceMotion":
      valid = (pose_calibrator.calib_valid and msg.angularVelocityDevice.valid and msg.orientationNED.valid and
               msg.inputsOK and msg.sensorsOK and msg.posenetOK)
      if valid:
        pose = Pose.from_device_motion(msg)
        calibrated_pose = pose_calibrator.build_calibrated_pose(pose)
        out.add("yaw_rate", t, calibrated_pose.angular_velocity.yaw)
        out.add("device_roll", t, pose.orientation.roll)

    elif which == "controlsState":
      out.add("control_curvature", t, msg.desiredCurvature)
      out.add("actual_curvature", t, msg.curvature)
      state = msg.lateralControlState
      if state.which() == "torqueState":
        torque = state.torqueState
        for name, attr in (
          ("pid_error", "error"), ("pid_p", "p"), ("pid_i", "i"), ("pid_d", "d"),
          ("pid_f", "f"), ("pid_output", "output"), ("desired_lat_accel", "desiredLateralAccel"),
          ("actual_lat_accel", "actualLateralAccel"), ("desired_lat_jerk", "desiredLateralJerk"),
          ("saturated", "saturated"), ("torque_version", "version"),
        ):
          out.add(name, t, getattr(torque, attr))

    elif which == "carState":
      for name, attr in (
        ("v_ego", "vEgo"), ("steering_angle_deg", "steeringAngleDeg"),
        ("steering_rate_deg", "steeringRateDeg"), ("driver_torque", "steeringTorque"),
        ("eps_torque", "steeringTorqueEps"), ("steering_pressed", "steeringPressed"),
      ):
        out.add(name, t, getattr(msg, attr))

    elif which == "carControl":
      out.add("lat_active", t, msg.latActive)
      out.add("cmd_torque", t, msg.actuators.torque)

    elif which == "carOutput":
      out.add("out_torque", t, msg.actuatorsOutput.torque)
      out.add("torque_output_can", t, msg.actuatorsOutput.torqueOutputCan)

    elif which == "lateralDelay":
      out.add("lateral_delay", t, msg.lateralDelay)
      out.add("lateral_delay_estimate", t, msg.lateralDelayEstimate)
      out.add("lateral_delay_std", t, msg.lateralDelayEstimateStd)
      out.add("delay_status", t, _enum_number(msg.status))

    elif which == "lateralTorqueParameters":
      out.add("lat_accel_factor", t, msg.latAccelFactorFiltered)
      out.add("lat_accel_offset", t, msg.latAccelOffsetFiltered)
      out.add("friction", t, msg.frictionCoefficientFiltered)
      out.add("torque_use_params", t, msg.useParams)

    elif which == "vehicleParameters":
      out.add("steer_ratio", t, msg.steerRatio)
      out.add("stiffness_factor", t, msg.stiffnessFactor)
      out.add("roll", t, msg.roll)
      out.add("angle_offset_deg", t, msg.angleOffsetDeg)

  return out


def _deduplicate(signal: Signal) -> tuple[np.ndarray, np.ndarray]:
  t = np.asarray(signal.times, dtype=np.float64)
  v = np.asarray(signal.values, dtype=np.float64)
  if len(t) == 0:
    return t, v
  order = np.argsort(t, kind="stable")
  t, v = t[order], v[order]
  keep = np.r_[np.diff(t) > 0, True]
  return t[keep], v[keep]


def interpolate_signal(signal: Signal | None, master_t: np.ndarray, discrete: bool = False) -> np.ndarray:
  if signal is None or not signal.times:
    return np.full_like(master_t, np.nan)
  t, v = _deduplicate(signal)
  if discrete:
    idx = np.searchsorted(t, master_t, side="right") - 1
    valid = (idx >= 0) & (master_t <= t[-1])
    result = np.full_like(master_t, np.nan)
    result[valid] = v[np.clip(idx[valid], 0, len(v) - 1)]
    return result
  result = np.interp(master_t, t, v, left=np.nan, right=np.nan)
  return result


def resample(extracted: ExtractedLog) -> dict[str, np.ndarray]:
  master = extracted.signals.get("control_curvature") or extracted.signals.get("cmd_torque")
  if master is None or len(master.times) < 2:
    raise ValueError("controlsState/carControl data is missing; an rlog is recommended")
  master_t, _ = _deduplicate(master)
  data = {"t": master_t}
  for name in CONTINUOUS_SIGNALS:
    data[name] = interpolate_signal(extracted.signals.get(name), master_t)
  for name in DISCRETE_SIGNALS:
    data[name] = interpolate_signal(extracted.signals.get(name), master_t, discrete=True)
  data["speed_kph"] = data["v_ego"] * 3.6
  data["model_lat_accel"] = data["model_curvature"] * data["v_ego"] ** 2
  data["control_lat_accel"] = data["control_curvature"] * data["v_ego"] ** 2
  data["inertial_lat_accel"] = data["v_ego"] * data["yaw_rate"]
  data["gravity_adjusted_inertial_lat_accel"] = (
    data["inertial_lat_accel"] - ACCELERATION_DUE_TO_GRAVITY * np.sin(data["device_roll"])
  )
  data["gravity_adjusted_vm_lat_accel"] = (
    data["actual_lat_accel"] - ACCELERATION_DUE_TO_GRAVITY * np.sin(data["roll"])
  )
  data["curvature_tracking_error"] = data["control_curvature"] - data["actual_curvature"]
  data["torque_limit_delta"] = data["cmd_torque"] - data["out_torque"]
  # With LateralJerkTorqueController enabled, pid_f is in normalized torque space:
  #   desired gravity-adjusted lateral accel / live factor + friction torque.
  # Subtract the linear part to isolate the friction/jerk contribution. This is
  # intentionally an estimate: use it only when the recorded runtime toggle is on.
  valid_factor = np.isfinite(data["lat_accel_factor"]) & (np.abs(data["lat_accel_factor"]) > 1e-6)
  base_jerk_aware_ff = np.full_like(master_t, np.nan)
  base_jerk_aware_ff[valid_factor] = (
    data["desired_lat_accel"][valid_factor] - ACCELERATION_DUE_TO_GRAVITY * data["roll"][valid_factor]
  ) / data["lat_accel_factor"][valid_factor]
  data["estimated_jerk_friction_torque"] = data["pid_f"] - base_jerk_aware_ff
  lat_accel_error = data["desired_lat_accel"] - data["actual_lat_accel"]
  data["estimated_patched_jerk_friction_torque"] = (
    data["friction"] * np.clip(
      LAT_ACCEL_FRICTION_FACTOR * lat_accel_error / JERK_AWARE_FRICTION_THRESHOLD,
      -1.0, 1.0,
    )
  )
  data["counterfactual_patched_jerk_cmd_torque"] = (
    data["cmd_torque"] + data["estimated_jerk_friction_torque"]
    - data["estimated_patched_jerk_friction_torque"]
  )
  # One-step (not closed-loop) comparison against V0's standard friction path.
  # Niro Plus has a zero offline steering-angle deadzone, so its normalized
  # standard-V0 friction torque is friction * clip(lat-accel error / 0.3).
  data["estimated_base_v0_friction_torque"] = (
    data["friction"] * np.clip(lat_accel_error / V0_FRICTION_THRESHOLD, -1.0, 1.0)
  )
  data["counterfactual_base_v0_cmd_torque"] = (
    data["cmd_torque"] + data["estimated_jerk_friction_torque"]
    - data["estimated_base_v0_friction_torque"]
  )
  offset_valid = valid_factor & np.isfinite(data["lat_accel_offset"])
  data["counterfactual_base_v0_cmd_torque"][offset_valid] += (
    data["lat_accel_offset"][offset_valid] / data["lat_accel_factor"][offset_valid]
  )
  return data


def contiguous_regions(mask: np.ndarray, min_samples: int) -> list[tuple[int, int]]:
  edges = np.diff(mask.astype(np.int8), prepend=0, append=0)
  starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
  return [(int(a), int(b)) for a, b in zip(starts, ends, strict=True) if b - a >= min_samples]


def select_non_overlapping_windows(windows: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
  selected: list[dict[str, Any]] = []
  for candidate in sorted(windows, key=lambda x: x["_score"], reverse=True):
    start, end = candidate["_startIndex"], candidate["_endIndex"]
    if any(start < kept["_endIndex"] and end > kept["_startIndex"] for kept in selected):
      continue
    selected.append(candidate)
    if len(selected) == top_n:
      break
  return selected


def band_component(x: np.ndarray, fs: float, band: tuple[float, float] = OSCILLATION_BAND_HZ) -> np.ndarray:
  x = np.asarray(x, dtype=np.float64)
  finite = np.isfinite(x)
  if len(x) < 4 or np.count_nonzero(finite) < max(4, math.ceil(0.8 * len(x))):
    return np.full_like(x, np.nan)
  if not np.all(finite):
    idx = np.arange(len(x))
    x = np.interp(idx, idx[finite], x[finite])
  x = x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))
  xf = np.fft.rfft(x)
  freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
  xf[(freqs < band[0]) | (freqs > min(band[1], 0.45 * fs))] = 0
  return np.fft.irfft(xf, n=len(x))


def spectral_metrics(x: np.ndarray, fs: float, target_hz: float | None = None) -> dict[str, float]:
  x = np.asarray(x, dtype=np.float64)
  finite = np.isfinite(x)
  if len(x) < 8 or np.count_nonzero(finite) < max(8, math.ceil(0.8 * len(x))):
    return {"dominantHz": math.nan, "dominantAmplitude": math.nan, "bandRms": math.nan,
            "bandPowerRatio": math.nan, "phaseDeg": math.nan}
  if not np.all(finite):
    idx = np.arange(len(x))
    x = np.interp(idx, idx[finite], x[finite])
  idx = np.arange(len(x))
  x = x - np.polyval(np.polyfit(idx, x, 1), idx)
  window = np.hanning(len(x))
  spectrum = np.fft.rfft(x * window)
  freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
  hi = min(OSCILLATION_BAND_HZ[1], 0.45 * fs)
  band = (freqs >= OSCILLATION_BAND_HZ[0]) & (freqs <= hi)
  total = (freqs >= 0.05) & (freqs <= hi)
  if not np.any(band):
    return {"dominantHz": math.nan, "dominantAmplitude": math.nan, "bandRms": math.nan,
            "bandPowerRatio": math.nan, "phaseDeg": math.nan}
  if target_hz is None:
    k = np.flatnonzero(band)[np.argmax(np.abs(spectrum[band]))]
  else:
    k = int(np.argmin(np.abs(freqs - target_hz)))
  amplitude = 2.0 * abs(spectrum[k]) / max(np.sum(window), np.finfo(float).eps)
  band_signal = band_component(x, fs)
  band_power = float(np.sum(np.abs(spectrum[band]) ** 2))
  total_power = float(np.sum(np.abs(spectrum[total]) ** 2))
  return {
    "dominantHz": float(freqs[k]),
    "dominantAmplitude": float(amplitude),
    "bandRms": float(np.sqrt(np.mean(band_signal ** 2))),
    "bandPowerRatio": band_power / total_power if total_power > 0 else math.nan,
    "phaseDeg": float(np.degrees(np.angle(spectrum[k]))),
  }


def best_lag(reference: np.ndarray, response: np.ndarray, fs: float, max_lag_s: float = 0.7) -> tuple[float, float]:
  """Return response delay (positive means response follows reference) and correlation."""
  x = band_component(reference, fs)
  y = band_component(response, fs)
  if len(x) < 8 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
    return math.nan, math.nan
  max_lag = min(int(round(max_lag_s * fs)), len(x) // 3)
  best = (math.nan, -math.inf)
  for lag in range(-max_lag, max_lag + 1):
    if lag >= 0:
      xa, ya = x[:len(x) - lag or None], y[lag:]
    else:
      xa, ya = x[-lag:], y[:len(y) + lag]
    if len(xa) < 4 or np.std(xa) == 0 or np.std(ya) == 0:
      continue
    corr = float(np.corrcoef(xa, ya)[0, 1])
    if corr > best[1]:
      best = (lag / fs, corr)
  return best


def _finite_fraction(condition: np.ndarray, valid: np.ndarray) -> float:
  count = int(np.count_nonzero(valid))
  return float(np.count_nonzero(condition & valid) / count) if count else math.nan


def _finite_stat(values: np.ndarray, fn: Any) -> float:
  values = values[np.isfinite(values)]
  return float(fn(values)) if len(values) else math.nan


def analyze_window(data: dict[str, np.ndarray], start: int, end: int, fs: float) -> dict[str, Any]:
  sl = slice(start, end)
  primary_name = "out_torque" if np.all(np.isfinite(data["out_torque"][sl])) else "cmd_torque"
  primary = spectral_metrics(data[primary_name][sl], fs)
  target_hz = primary["dominantHz"]
  names = (
    "model_lat_accel", "control_lat_accel", "desired_lat_accel", "actual_lat_accel", "inertial_lat_accel",
    "gravity_adjusted_inertial_lat_accel", "gravity_adjusted_vm_lat_accel",
    "pid_error", "pid_p", "pid_i", "pid_f", "cmd_torque", "out_torque",
    "estimated_jerk_friction_torque", "estimated_base_v0_friction_torque",
    "estimated_patched_jerk_friction_torque", "counterfactual_patched_jerk_cmd_torque",
    "counterfactual_base_v0_cmd_torque", "steering_angle_deg", "steering_rate_deg", "driver_torque",
  )
  spectra = {name: spectral_metrics(data[name][sl], fs, target_hz) for name in names}
  pairs = {
    "modelToControl": ("model_lat_accel", "control_lat_accel"),
    "desiredToActualLatAccel": ("desired_lat_accel", "actual_lat_accel"),
    "vehicleModelToInertialLatAccel": ("actual_lat_accel", "inertial_lat_accel"),
    "commandToOutputTorque": ("cmd_torque", "out_torque"),
    "outputTorqueToSteeringRate": ("out_torque", "steering_rate_deg"),
  }
  lag = {}
  for label, (a, b) in pairs.items():
    delay, corr = best_lag(data[a][sl], data[b][sl], fs)
    lag[label] = {"responseDelaySec": delay, "correlation": corr}

  cmd, actual = data["cmd_torque"][sl], data["out_torque"][sl]
  valid_torque = np.isfinite(cmd) & np.isfinite(actual)
  saturated = data["saturated"][sl]
  valid_sat = np.isfinite(saturated)
  output_band = band_component(actual, fs)
  zero_crossings = np.count_nonzero(np.diff(np.signbit(output_band))) if np.all(np.isfinite(output_band)) else 0

  desired_rms = spectra["desired_lat_accel"]["bandRms"]
  actual_rms = spectra["actual_lat_accel"]["bandRms"]
  inertial_rms = spectra["inertial_lat_accel"]["bandRms"]
  feedback_amplification = actual_rms / desired_rms if desired_rms and math.isfinite(desired_rms) else math.nan
  vehicle_model_inertial_gain = inertial_rms / actual_rms \
    if actual_rms and math.isfinite(actual_rms) and math.isfinite(inertial_rms) else math.nan
  pif_total = sum(spectra[name]["bandRms"] ** 2 for name in ("pid_p", "pid_i", "pid_f")
                  if math.isfinite(spectra[name]["bandRms"]))
  feedforward_share = spectra["pid_f"]["bandRms"] ** 2 / pif_total if pif_total > 0 else math.nan
  feedback_terms = ("pid_p", "pid_i", "estimated_jerk_friction_torque")
  feedback_total = sum(spectra[name]["bandRms"] ** 2 for name in feedback_terms
                       if math.isfinite(spectra[name]["bandRms"]))
  friction_rms = spectra["estimated_jerk_friction_torque"]["bandRms"]
  estimated_friction_share = friction_rms ** 2 / feedback_total if feedback_total > 0 and math.isfinite(friction_rms) else math.nan
  logged_cmd_rms = spectra["cmd_torque"]["bandRms"]
  counterfactual_cmd_rms = spectra["counterfactual_base_v0_cmd_torque"]["bandRms"]
  counterfactual_ratio = counterfactual_cmd_rms / logged_cmd_rms \
    if logged_cmd_rms and math.isfinite(logged_cmd_rms) and math.isfinite(counterfactual_cmd_rms) else math.nan
  patched_cmd_rms = spectra["counterfactual_patched_jerk_cmd_torque"]["bandRms"]
  patched_ratio = patched_cmd_rms / logged_cmd_rms \
    if logged_cmd_rms and math.isfinite(logged_cmd_rms) and math.isfinite(patched_cmd_rms) else math.nan

  return {
    "startMonoTimeSec": float(data["t"][start]),
    "endMonoTimeSec": float(data["t"][end - 1]),
    "startSec": float(data["t"][start] - data["t"][0]),
    "endSec": float(data["t"][end - 1] - data["t"][0]),
    "durationSec": float(data["t"][end - 1] - data["t"][start]),
    "meanSpeedKph": float(np.nanmean(data["speed_kph"][sl])),
    "dominantHz": target_hz,
    "torqueSignReversalsPerSec": float(zero_crossings / max((end - start) / fs, 1e-6)),
    "commandOutputMismatchFraction": _finite_fraction(np.abs(cmd - actual) > 0.01, valid_torque),
    "saturationFraction": _finite_fraction(saturated > 0.5, valid_sat),
    "feedbackAmplification": feedback_amplification,
    "vehicleModelInertialBandGain": vehicle_model_inertial_gain,
    "feedforwardBandPowerShare": feedforward_share,
    "estimatedJerkFrictionBandPowerShare": estimated_friction_share,
    "oneStepPatchedJerkCommandBandRmsRatio": patched_ratio,
    "oneStepBaseV0CommandBandRmsRatio": counterfactual_ratio,
    "modelTiming": {
      "executionTimeMeanSec": _finite_stat(data["model_execution_time"][sl], np.mean),
      "executionTimeMaxSec": _finite_stat(data["model_execution_time"][sl], np.max),
      "frameDropPercentMean": _finite_stat(data["frame_drop_percent"][sl], np.mean),
      "frameAgeMax": _finite_stat(data["frame_age"][sl], np.max),
      "cameraDroppedFraction": _finite_fraction(data["camera_frame_dropped"][sl] > 0.5,
                                                 np.isfinite(data["camera_frame_dropped"][sl])),
    },
    "spectra": spectra,
    "lag": lag,
    "_startIndex": start,
    "_endIndex": end,
    "_score": float(spectra["out_torque"]["bandRms"] if math.isfinite(spectra["out_torque"]["bandRms"])
                    else spectra["cmd_torque"]["bandRms"]),
  }


def parameter_summary(extracted: ExtractedLog, data: dict[str, np.ndarray]) -> dict[str, Any]:
  summary: dict[str, Any] = {
    "carParams": extracted.car_params,
    "carParamsSP": extracted.car_params_sp,
    "recordedParams": extracted.init_params,
    "git": extracted.git,
  }
  for name in ("torque_version", "torque_use_params", "lat_accel_factor", "lat_accel_offset", "friction",
               "lateral_delay", "lateral_delay_estimate", "lateral_delay_std", "steer_ratio", "stiffness_factor"):
    vals = data[name]
    vals = vals[np.isfinite(vals)]
    if len(vals):
      summary[name] = {"min": float(np.min(vals)), "median": float(np.median(vals)), "max": float(np.max(vals))}

  lagd_toggle = extracted.init_params.get("LagdToggle")
  sw_delay = extracted.init_params.get("LagdToggleDelay")
  actuator_delay = extracted.car_params.get("steerActuatorDelay")
  if lagd_toggle == "0" and sw_delay is not None and actuator_delay is not None and "lateral_delay" in summary:
    try:
      fixed = float(actuator_delay) + float(sw_delay)
      raw = summary["lateral_delay"]["median"]
      runner = (extracted.init_params.get("ModelManager_ActiveBundle.runner") or
                extracted.init_params.get("ModelManager_ActiveBundleChestnut.runner"))
      summary["delayPathCheck"] = {
        "liveLearningEnabled": False,
        "activeBundleRunner": runner,
        "modelFixedDelaySec": fixed,
        "controllerRawDelaySec": raw,
        "mismatchSec": raw - fixed,
        "potentialModelControllerMismatch": runner not in (None, "stock"),
        "note": "controlsd passes raw lateralDelay to V0. modeld_v2 uses the fixed cache when LagdToggle is off; the stock modeld path currently uses raw delay for its action horizon.",
      }
    except ValueError:
      pass
  return summary


def analyze(data: dict[str, np.ndarray], extracted: ExtractedLog, speed_min_kph: float, window_sec: float,
            top_n: int) -> dict[str, Any]:
  dt = np.diff(data["t"])
  dt = dt[(dt > 0) & (dt < 0.2)]
  if not len(dt):
    raise ValueError("unable to infer control sample rate")
  fs = 1.0 / float(np.median(dt))
  valid = np.isfinite(data["speed_kph"]) & (data["speed_kph"] >= speed_min_kph)
  if np.any(np.isfinite(data["lat_active"])):
    valid &= data["lat_active"] > 0.5
  if np.any(np.isfinite(data["steering_pressed"])):
    valid &= data["steering_pressed"] < 0.5
  # Never construct an FFT window across missing route segments or corrupt-log gaps.
  discontinuity = np.r_[False, np.diff(data["t"]) > max(0.2, 3.0 / fs)]
  valid[discontinuity] = False
  samples = max(8, int(round(window_sec * fs)))
  regions = contiguous_regions(valid, samples)
  windows: list[dict[str, Any]] = []
  hop = max(1, samples // 2)
  for region_start, region_end in regions:
    starts = list(range(region_start, region_end - samples + 1, hop))
    tail = region_end - samples
    if tail >= region_start and (not starts or starts[-1] != tail):
      starts.append(tail)
    for start in starts:
      windows.append(analyze_window(data, start, start + samples, fs))
  windows = select_non_overlapping_windows(windows, top_n)
  return {
    "schemaVersion": 1,
    "readOnlyDiagnostic": True,
    "inputs": [str(p) for p in []],
    "sampleRateHz": fs,
    "oscillationBandHz": list(OSCILLATION_BAND_HZ),
    "speedThresholdKph": speed_min_kph,
    "windowSec": window_sec,
    "eligibleDurationSec": float(np.count_nonzero(valid) / fs),
    "highSpeedWindowsFound": len(windows),
    "runtime": parameter_summary(extracted, data),
    "messageCounts": extracted.message_counts,
    "windows": windows,
  }


def _clean_for_json(value: Any) -> Any:
  if isinstance(value, dict):
    return {k: _clean_for_json(v) for k, v in value.items() if not k.startswith("_")}
  if isinstance(value, list):
    return [_clean_for_json(v) for v in value]
  if isinstance(value, float) and not math.isfinite(value):
    return None
  return value


def write_csv(path: Path, data: dict[str, np.ndarray], windows: list[dict[str, Any]]) -> None:
  names = ["t", "speed_kph", "lat_active", "steering_pressed", "model_lat_accel", "control_lat_accel",
           "desired_lat_accel", "actual_lat_accel", "inertial_lat_accel", "gravity_adjusted_inertial_lat_accel",
           "gravity_adjusted_vm_lat_accel", "pid_error", "pid_p", "pid_i", "pid_f", "cmd_torque",
           "out_torque", "torque_output_can", "steering_angle_deg", "steering_rate_deg", "driver_torque",
           "saturated", "lateral_delay", "lat_accel_factor", "lat_accel_offset", "friction",
           "estimated_jerk_friction_torque", "estimated_base_v0_friction_torque",
           "estimated_patched_jerk_friction_torque", "counterfactual_patched_jerk_cmd_torque",
           "counterfactual_base_v0_cmd_torque", "model_execution_time",
           "frame_drop_percent", "frame_age", "model_big", "camera_frame_dropped"]
  selected = np.zeros(len(data["t"]), dtype=bool)
  for window in windows:
    selected[window["_startIndex"]:window["_endIndex"]] = True
  with path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(names)
    for i in np.flatnonzero(selected):
      writer.writerow([f"{data[name][i]:.9g}" if math.isfinite(data[name][i]) else "" for name in names])


def _polyline(values: np.ndarray, width: int, height: int, x0: int, y0: int, lo: float, hi: float) -> str:
  valid = np.isfinite(values)
  if np.count_nonzero(valid) < 2:
    return ""
  x = np.linspace(x0, x0 + width, len(values))
  y = y0 + height - (np.clip(values, lo, hi) - lo) / (hi - lo) * height
  return " ".join(f"{a:.1f},{b:.1f}" for a, b, ok in zip(x, y, valid, strict=True) if ok)


def _svg_panel(data: dict[str, np.ndarray], window: dict[str, Any], series: list[tuple[str, str, str]], title: str,
               shared_scale: bool = True) -> str:
  start, end = window["_startIndex"], window["_endIndex"]
  width, height, x0, y0 = 920, 150, 55, 25
  parts = [f'<svg viewBox="0 0 1000 205" role="img" aria-label="{html.escape(title)}">',
           f'<text x="10" y="16">{html.escape(title)}</text>',
           f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" fill="none" stroke="#999"/>']
  arrays = [data[name][start:end] for name, _, _ in series]
  finite = np.concatenate([a[np.isfinite(a)] for a in arrays])
  common_lo, common_hi = np.percentile(finite, [2, 98]) if len(finite) else (-1.0, 1.0)
  if math.isclose(common_lo, common_hi):
    common_lo, common_hi = common_lo - 1.0, common_hi + 1.0
  for (name, color, label), values in zip(series, arrays, strict=True):
    finite_values = values[np.isfinite(values)]
    if len(finite_values) < 2:
      continue
    lo, hi = (common_lo, common_hi) if shared_scale else np.percentile(finite_values, [2, 98])
    if math.isclose(lo, hi):
      lo, hi = lo - 1.0, hi + 1.0
    points = _polyline(values, width, height, x0, y0, lo, hi)
    if points:
      parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.4"/>')
  legend_x = 60
  for _, color, label in series:
    parts.append(f'<text x="{legend_x}" y="195" fill="{color}">{html.escape(label)}</text>')
    legend_x += 190
  parts.append("</svg>")
  return "".join(parts)


def write_html(path: Path, report: dict[str, Any], data: dict[str, np.ndarray]) -> None:
  runtime_json = html.escape(json.dumps(_clean_for_json(report["runtime"]), ensure_ascii=False, indent=2))
  blocks = []
  for i, window in enumerate(report["windows"], 1):
    blocks.append(f"<h2>#{i}: {window['startSec']:.1f}–{window['endSec']:.1f}s, "
                  f"{window['meanSpeedKph']:.1f} km/h, {window['dominantHz']:.2f} Hz</h2>")
    blocks.append(_svg_panel(data, window, [("model_lat_accel", "#377eb8", "model"),
                                                  ("desired_lat_accel", "#ff7f00", "controller desired"),
                                                  ("actual_lat_accel", "#4daf4a", "actual")],
                                  "Lateral acceleration (m/s², shared scale)"))
    blocks.append(_svg_panel(data, window, [("pid_p", "#e41a1c", "P"), ("pid_i", "#984ea3", "I"),
                                                  ("pid_f", "#4daf4a", "FF"),
                                                  ("estimated_jerk_friction_torque", "#ff7f00", "estimated jerk friction")],
                                  "V0/Jerk-aware terms (shared scale)"))
    blocks.append(_svg_panel(data, window, [("cmd_torque", "#377eb8", "requested"),
                                                  ("out_torque", "#e41a1c", "after car controller")],
                                  "Normalized steering torque (shared scale)"))
    blocks.append(_svg_panel(data, window, [("cmd_torque", "#377eb8", "logged jerk-aware request"),
                                                  ("counterfactual_patched_jerk_cmd_torque", "#ff7f00", "patched 0.7 estimate"),
                                                  ("counterfactual_base_v0_cmd_torque", "#4daf4a", "one-step base V0 estimate")],
                                  "Offline request comparison (not a closed-loop replay)"))
    blocks.append(_svg_panel(data, window, [("steering_angle_deg", "#984ea3", "angle"),
                                                  ("steering_rate_deg", "#ff7f00", "rate")],
                                  "Steering response (angle/rate individually scaled)", shared_scale=False))
    public = _clean_for_json(window)
    blocks.append(f"<details><summary>window metrics</summary><pre>{html.escape(json.dumps(public, indent=2))}</pre></details>")
  body = "\n".join(blocks) if blocks else "<p>No eligible high-speed active-control window was found.</p>"
  path.write_text(f"""<!doctype html><meta charset="utf-8"><title>Lateral oscillation diagnostic</title>
<style>body{{font:14px system-ui;margin:24px;max-width:1100px}}svg{{width:100%;background:#fafafa;margin:6px 0}}pre{{overflow:auto;background:#f4f4f4;padding:12px}}h2{{margin-top:36px}}</style>
<h1>Read-only lateral oscillation diagnostic</h1>
<p>Band: {report['oscillationBandHz'][0]}–{report['oscillationBandHz'][1]} Hz; minimum speed: {report['speedThresholdKph']} km/h; eligible time: {report['eligibleDurationSec']:.1f}s.</p>
<p>Traces are for diagnosis only. Panels use a shared scale except the mixed-unit steering angle/rate panel.</p>
<h2>Runtime snapshot</h2><pre>{runtime_json}</pre>{body}""", encoding="utf-8")


def resolve_inputs(inputs: list[str]) -> list[Path]:
  paths: list[Path] = []
  for raw in inputs:
    path = Path(raw).expanduser()
    if path.is_dir():
      rlogs = sorted(path.rglob("rlog.zst"))
      paths.extend(rlogs if rlogs else sorted(path.rglob("qlog.zst")))
    elif path.is_file():
      paths.append(path)
    else:
      raise FileNotFoundError(raw)
  unique = list(dict.fromkeys(p.resolve() for p in paths))
  if not unique:
    raise FileNotFoundError("no rlog.zst/qlog.zst files found")
  return unique


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("logs", nargs="+", help="rlog/qlog file(s), or a directory containing them")
  parser.add_argument("--output-dir", type=Path, default=Path("lateral_oscillation_report"))
  parser.add_argument("--speed-min-kph", type=float, default=80.0)
  parser.add_argument("--window-sec", type=float, default=8.0)
  parser.add_argument("--top", type=int, default=8)
  args = parser.parse_args()

  paths = resolve_inputs(args.logs)
  extracted = extract_logs(paths)
  data = resample(extracted)
  report = analyze(data, extracted, args.speed_min_kph, args.window_sec, args.top)
  report["inputs"] = [str(path) for path in paths]

  args.output_dir.mkdir(parents=True, exist_ok=True)
  write_csv(args.output_dir / "top_windows.csv", data, report["windows"])
  write_html(args.output_dir / "report.html", report, data)
  (args.output_dir / "report.json").write_text(
    json.dumps(_clean_for_json(report), ensure_ascii=False, indent=2), encoding="utf-8")
  print(f"Wrote {args.output_dir / 'report.html'}")
  print(f"Eligible high-speed active-control time: {report['eligibleDurationSec']:.1f}s")
  print(f"Ranked windows: {report['highSpeedWindowsFound']}")


if __name__ == "__main__":
  main()
