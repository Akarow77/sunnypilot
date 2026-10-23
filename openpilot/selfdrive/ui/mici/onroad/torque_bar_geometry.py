TORQUE_ANGLE_SPAN = 12.7
TORQUE_REFERENCE_WIDTH = 476


def torque_angle_span(view_width: float, scale: float) -> float:
  """Shrink beside navigation, without enlarging the arc on the scaled 3X UI."""
  width_ratio = max(0.0, view_width) / (TORQUE_REFERENCE_WIDTH * scale)
  return TORQUE_ANGLE_SPAN * min(width_ratio, 1.0)
