"""Narrow LAN GPU exceptions; sensor, timing and transport failures remain errors."""
import math

LAN_GPU_MAX_FRAME_DROP = 10


def model_lagging(frame_drop_percent: float, lan_active: bool) -> bool:
  return not math.isfinite(frame_drop_percent) or frame_drop_percent > (LAN_GPU_MAX_FRAME_DROP if lan_active else 1)


def is_lan_pose_drop(pose, valid: bool, lan_active: bool) -> bool:
  if not lan_active or valid or not pose.frameDropped:
    return False
  # A drop marker is not permission to disregard corrupt measurements.
  for field in ("trans", "rot", "wideFromDeviceEuler", "roadTransformTrans"):
    values, stds = getattr(pose, field), getattr(pose, field + "Std")
    if len(values) != 3 or len(stds) != 3 or not all(math.isfinite(v) for v in values):
      return False
    if not all(math.isfinite(v) and v > 0 for v in stds):
      return False
  return True
