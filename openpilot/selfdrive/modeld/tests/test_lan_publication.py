"""Validate the LAN outputs through real native action and Cap'n Proto writers."""
import json

import numpy as np
import pytest

from openpilot.cereal import log
from openpilot.selfdrive.modeld import fill_model_msg as writers
from openpilot.selfdrive.modeld import lan_model
from openpilot.selfdrive.modeld.gpu_on_lan import validate_metadata
from openpilot.selfdrive.modeld.modeld import get_action_from_model
from openpilot.selfdrive.modeld.tests.test_gpu_on_lan import metadata
from openpilot.selfdrive.modeld.tests.test_lan_model import prepared_model, run


@pytest.mark.parametrize('raw_predictions', [False, True])
@pytest.mark.parametrize('action_width,lead_width', [(2, 144), (4, 102), (None, 144)])
def test_lan_native_messages_serialize(monkeypatch, raw_predictions, action_width, lead_width):
  monkeypatch.setattr(lan_model, 'SEND_RAW_PRED', raw_predictions)
  monkeypatch.setattr(writers, 'SEND_RAW_PRED', raw_predictions)
  meta = metadata()
  slices = json.loads(meta['parameters']['output_slices'])
  if action_width is None:
    del slices['action']  # Native action can also be derived from the plan.
  else:
    slices['action'][1] = slices['action'][0] + action_width
  slices['lead'][1] = slices['lead'][0] + lead_width
  meta['parameters']['output_slices'] = json.dumps(slices)
  _, spans, size = validate_metadata(meta)
  model, client = prepared_model()
  client.output_slices = spans
  outputs = run(model, client, np.zeros(8, np.float32))
  action = get_action_from_model(outputs, log.ModelDataV2.Action(), .2, .5, 15.)
  assert np.isfinite([action.desiredCurvature, action.desiredAcceleration]).all()

  model_msg = log.Event.new_message()
  model_msg.init('modelV2')
  writers.fill_model_msg(model_msg, outputs, action, writers.PublishState(), 40, 40, 40, 0., 123456789, .05, True)
  model_msg.modelV2.big = True
  driving_msg = log.Event.new_message()
  driving_msg.init('drivingModelData')
  writers.fill_driving_model_data(driving_msg, model_msg)
  pose_msg = log.Event.new_message()
  pose_msg.init('cameraOdometry')
  writers.fill_pose_msg(pose_msg, outputs, 40, 0, 123456789, True)

  for msg in (model_msg, driving_msg, pose_msg):
    with log.Event.from_bytes(msg.to_bytes()) as decoded:
      assert decoded.valid
      assert getattr(decoded, decoded.which()).frameId == 40
  assert model_msg.modelV2.big and driving_msg.drivingModelData.big
  assert len(model_msg.modelV2.position.x) == 33
  assert len(model_msg.modelV2.leadsV3) == 3
  assert len(pose_msg.cameraOdometry.trans) == 3
  assert np.all(np.asarray(pose_msg.cameraOdometry.transStd) > 0)
  assert len(model_msg.modelV2.rawPredictions) == (size * 4 if raw_predictions else 0)
