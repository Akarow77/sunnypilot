import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from openpilot.common.lan_gpu import is_lan_pose_drop, model_lagging


@pytest.mark.parametrize('active,rate,lagging', [(False, 1, False), (False, 2, True), (True, 2, False),
                                               (True, 10, False), (True, 10.1, True), (True, float('nan'), True)])
def test_lag_threshold(active, rate, lagging):
  assert model_lagging(rate, active) == lagging


def pose():
  p = SimpleNamespace(frameDropped=True)
  for field in ('trans', 'rot', 'wideFromDeviceEuler', 'roadTransformTrans'):
    setattr(p, field, [0., 0., 0.])
    setattr(p, field + 'Std', [1., 1., 1.])
  return p


def test_only_explicit_drop_can_be_exempted():
  p = pose()
  assert is_lan_pose_drop(p, False, True)
  assert not is_lan_pose_drop(p, True, True)
  assert not is_lan_pose_drop(p, False, False)
  p.frameDropped = False
  assert not is_lan_pose_drop(p, False, True)


def test_drop_marker_schema_roundtrip():
  from openpilot.cereal import log
  msg = log.Event.new_message()
  msg.init('cameraOdometry')
  assert msg.cameraOdometry.frameDropped is False
  msg.cameraOdometry.frameDropped = True
  with log.Event.from_bytes(msg.to_bytes()) as decoded:
    assert decoded.cameraOdometry.frameDropped is True


@pytest.mark.parametrize('field,value', [('trans', [float('nan'), 0, 0]), ('rotStd', [-1, 1, 1]),
                                       ('roadTransformTransStd', [0, 1, 1]), ('wideFromDeviceEuler', []),
                                       ('rot', [float('inf'), 0, 0])])
def test_corrupt_pose_is_not_a_drop_exception(field, value):
  p = pose()
  setattr(p, field, value)
  assert not is_lan_pose_drop(p, False, True)


@pytest.mark.parametrize('relative', ['selfdrive/modeld/fill_model_msg.py', 'sunnypilot/modeld_v2/fill_model_msg.py'])
@pytest.mark.parametrize('calibrated,drops,marked', [(False, 0, False), (False, 2, False), (True, 0, False), (True, 1, True)])
def test_drop_marker_requires_calibration(relative, calibrated, drops, marked):
  # Exercise the real marker expression without importing device-only model dependencies.
  root = Path(__file__).resolve().parents[2]
  tree = ast.parse((root / relative).read_text())
  fill = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'fill_pose_msg')
  assign = next(node for node in ast.walk(fill) if isinstance(node, ast.Assign) and
                any(isinstance(t, ast.Attribute) and t.attr == 'frameDropped' for t in node.targets))
  env = {'extrinsics_calibration_seen': calibrated, 'live_calib_seen': calibrated, 'vipc_dropped_frames': drops}
  assert eval(compile(ast.Expression(assign.value), '<drop marker>', 'eval'), env) == marked
