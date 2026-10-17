import math

from opendbc.car.car_helpers import interfaces
from opendbc.car.honda.values import CAR as HONDA
from opendbc.car.hyundai.values import CAR as HYUNDAI

from openpilot.common.params import Params
from openpilot.common.realtime import DT_CTRL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.car.helpers import convert_to_capnp
from openpilot.sunnypilot.selfdrive.car import interfaces as sunnypilot_interfaces
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_v0 import KP, LatControlTorque


def make_controller(car_name):
  params = Params()
  params.put_bool("EnforceTorqueControl", True, block=True)
  params.put_bool("LateralJerkTorqueController", False, block=True)
  params.put_bool("NeuralNetworkLateralControl", False, block=True)

  CarInterface = interfaces[car_name]
  CP = CarInterface.get_non_essential_params(car_name)
  CP_SP = CarInterface.get_non_essential_params_sp(CP, car_name)
  CI = CarInterface(CP, CP_SP)
  sunnypilot_interfaces.setup_interfaces(CI, params)
  CP_SP = convert_to_capnp(CP_SP)
  return LatControlTorque(CP.as_reader(), CP_SP.as_reader(), CI, DT_CTRL)


class TestLatControlTorqueV0(OpenpilotTestCase):
  def test_default_high_speed_kp(self):
    controller = make_controller(HONDA.HONDA_CIVIC)
    assert controller.high_speed_kp == KP
    controller.pid.speed = 30.0
    assert controller.pid.k_p == KP

  def test_niro_plus_high_speed_kp(self):
    controller = make_controller(HYUNDAI.KIA_NIRO_PLUS_EV)
    assert math.isclose(controller.high_speed_kp, 0.6, rel_tol=1e-6)
    controller.pid.speed = 30.0
    assert math.isclose(controller.pid.k_p, 0.6, rel_tol=1e-6)
