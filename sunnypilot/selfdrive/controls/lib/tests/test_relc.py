"""
Copyright (c) 2021-, rav4kumar, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import pytest

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
from openpilot.sunnypilot.selfdrive.controls.lib.relc import (
  RoadEdgeLaneChangeController, EDGE_REACTION_TIME, EDGE_CLEAR_TIME, MIN_SPEED,
)

V_HIGH = MIN_SPEED + 2.0
V_LOW = MIN_SPEED - 1.0


class DummyRoadEdge:
  def __init__(self, y):
    self.y = [y]


@pytest.fixture
def relc(mocker):
  mock_params = mocker.patch("openpilot.sunnypilot.selfdrive.controls.lib.relc.Params")
  mock_params.return_value.get_bool.return_value = True
  controller = RoadEdgeLaneChangeController(DesireHelper())
  controller.enabled = True
  return controller


def make_road_edges(left_y=-3.0, right_y=3.0):
  return [DummyRoadEdge(left_y), DummyRoadEdge(right_y)]


def drive(controller, road_edge_stds, lane_line_probs, seconds, v_ego=V_HIGH, road_edges=None):
  for _ in range(int(seconds / DT_MDL) + 1):
    controller.update(road_edge_stds, lane_line_probs, v_ego, road_edges)


@pytest.mark.parametrize("road_edge_stds,lane_line_probs,attr", [
  ([0.0, 0.9], [0.8, 0.0, 0.8, 0.8], "left_edge_detected"),
  ([0.9, 0.0], [0.8, 0.8, 0.0, 0.8], "right_edge_detected"),
])
def test_edge_detection(relc, road_edge_stds, lane_line_probs, attr):
  drive(relc, road_edge_stds, lane_line_probs, EDGE_REACTION_TIME + 0.1)
  assert getattr(relc, attr)


def test_edge_detection_requires_time(relc):
  drive(relc, [0.0, 0.9], [0.8, 0.0, 0.8, 0.8], EDGE_REACTION_TIME - 0.05)
  assert not relc.left_edge_detected


def test_both_edges_detected(relc):
  drive(relc, [0.0, 0.0], [0.8, 0.0, 0.0, 0.8], EDGE_REACTION_TIME + 0.1)
  assert relc.left_edge_detected
  assert relc.right_edge_detected


def test_noise_doesnt_clear(relc):
  edge = ([0.0, 0.9], [0.8, 0.0, 0.8, 0.8])
  clear = ([0.9, 0.9], [0.8, 0.8, 0.8, 0.8])

  drive(relc, *edge, EDGE_REACTION_TIME + 0.1)
  assert relc.left_edge_detected

  relc.update(*clear, V_HIGH)
  relc.update(*edge, V_HIGH)
  assert relc.left_edge_detected


def test_clears_after_window(relc):
  edge = ([0.0, 0.9], [0.8, 0.0, 0.8, 0.8])
  clear = ([0.9, 0.9], [0.8, 0.8, 0.8, 0.8])

  drive(relc, *edge, EDGE_REACTION_TIME + 0.1)
  assert relc.left_edge_detected

  drive(relc, *clear, EDGE_CLEAR_TIME + 0.05)
  assert not relc.left_edge_detected
  assert relc.left_edge_timer == 0.0


def test_low_speed_skips(relc):
  drive(relc, [0.0, 0.9], [0.8, 0.0, 0.8, 0.8], EDGE_REACTION_TIME + 0.1, v_ego=V_LOW)
  assert not relc.left_edge_detected
  assert relc.left_edge_timer == 0.0


def test_speed_drop_resets(relc):
  drive(relc, [0.0, 0.9], [0.8, 0.0, 0.8, 0.8], EDGE_REACTION_TIME + 0.1)
  assert relc.left_edge_detected

  relc.update([0.0, 0.9], [0.8, 0.0, 0.8, 0.8], V_LOW)
  assert not relc.left_edge_detected


def test_param_off_resets(relc):
  drive(relc, [0.0, 0.9], [0.8, 0.0, 0.8, 0.8], EDGE_REACTION_TIME + 0.1)
  assert relc.left_edge_detected

  relc.params.get_bool.return_value = False
  relc.read_params()
  relc.update([0.0, 0.9], [0.8, 0.0, 0.8, 0.8], V_HIGH)
  assert not relc.left_edge_detected
  assert not relc.right_edge_detected


@pytest.mark.parametrize("lane_line_probs", [
  [0.0, 0.8, 0.8, 0.8],
  [0.8, 0.8, 0.8, 0.0],
])
def test_outer_lane_lines_do_not_drive_edge_detection(relc, lane_line_probs):
  drive(relc, [0.0, 0.0], lane_line_probs, EDGE_REACTION_TIME + 0.1)
  assert not relc.left_edge_detected
  assert not relc.right_edge_detected


@pytest.mark.parametrize("road_edge_stds,road_edges,attr", [
  ([0.0, 0.9], make_road_edges(left_y=-3.0, right_y=8.0), "left_edge_detected"),
  ([0.9, 0.0], make_road_edges(left_y=-8.0, right_y=3.0), "right_edge_detected"),
])
def test_near_road_edge_geometry_blocks_with_visible_lane_lines(relc, road_edge_stds, road_edges, attr):
  drive(relc, road_edge_stds, [0.8, 0.8, 0.8, 0.8], EDGE_REACTION_TIME + 0.1, road_edges=road_edges)
  assert getattr(relc, attr)


def test_far_road_edge_geometry_does_not_block(relc):
  drive(relc, [0.0, 0.0], [0.8, 0.0, 0.0, 0.8], EDGE_REACTION_TIME + 0.1, road_edges=make_road_edges(left_y=-8.0, right_y=8.0))
  assert not relc.left_edge_detected
  assert not relc.right_edge_detected
