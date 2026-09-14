#!/usr/bin/env python3

import unittest

import numpy as np

try:
  from coreml_inference_server import CoreMLPolicySession
except ModuleNotFoundError as error:
  if error.name != 'coremltools':
    raise
  CoreMLPolicySession = None
from transport import POLICY_INPUTS, WARPED_BYTES, WARPED_SHAPE


class FakeModel:
  def __init__(self):
    self.calls = []

  def predict(self, inputs):
    self.calls.append({name: value.copy() for name, value in inputs.items()})
    output = np.zeros((1, 8), dtype=np.float16)
    output[:, 2:4] = len(self.calls)
    return {'output': output}


@unittest.skipIf(CoreMLPolicySession is None, 'coremltools is installed only in .coreml-venv')
class CoreMLPolicySessionTest(unittest.TestCase):
  def test_temporal_queues_match_frame_skip(self) -> None:
    metadata = {
      'input_shapes': {
        'img': (1, 12, 128, 256),
        'big_img': (1, 12, 128, 256),
        'desire_pulse': (1, 3, 8),
        'traffic_convention': (1, 2),
        'action_t': (1, 2),
        'features_buffer': (1, 2, 1, 2),
      },
      'output_shapes': {'outputs': (1, 8)},
      'output_slices': {'hidden_state': slice(2, 4)},
    }
    model = FakeModel()
    session = CoreMLPolicySession(model, metadata, 'output', frame_skip=2)
    warped = np.zeros(WARPED_SHAPE, dtype=np.uint8)
    warped[0] = 7
    warped[1] = 9
    policy = POLICY_INPUTS.pack(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.1, 0.1)
    payload = warped.tobytes() + policy
    self.assertEqual(len(payload), WARPED_BYTES + POLICY_INPUTS.size)

    session.infer(payload)
    session.infer(payload)
    session.infer(payload)
    first, second, third = model.calls
    self.assertTrue(np.all(first['img'][:, :6] == 0))
    self.assertTrue(np.all(first['img'][:, 6:] == 7))
    self.assertTrue(np.all(first['big_img'][:, 6:] == 9))
    self.assertEqual(first['desire_pulse'][0, -1, 0], 1.0)
    self.assertTrue(np.all(first['features_buffer'] == 0))
    self.assertTrue(np.all(second['features_buffer'] == 0))
    self.assertTrue(np.all(third['features_buffer'][:, -1] == 1.0))

    session.reset()
    session.infer(payload)
    reset_call = model.calls[-1]
    self.assertTrue(np.all(reset_call['img'][:, :6] == 0))
    self.assertTrue(np.all(reset_call['features_buffer'] == 0))
    self.assertEqual(reset_call['desire_pulse'][0, -1, 0], 1.0)


if __name__ == '__main__':
  unittest.main()
