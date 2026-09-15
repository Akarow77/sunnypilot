import json
from pathlib import Path
import tempfile
import unittest

from summarize_live_shadow import summarize_log


class ShadowSummaryTest(unittest.TestCase):
  def test_failure_and_unqualified_differences_are_not_hidden(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'live.jsonl'
      row = {'copyToOutputMs': 60., 'roundTripMs': 58., 'inferenceMs': 40.,
             'deadlineMiss': True, 'qualified': False, 'curvatureDifference': 99.}
      path.write_text(json.dumps(row) + '\n' + json.dumps({'error': 'timed out'}) + '\n{"incomplete":')
      report = summarize_log(path)
      self.assertEqual(report['completedFrameDeadlineMisses'], 1)
      self.assertEqual(report['timeoutOrDeadlineFailures'], 1)
      self.assertEqual(report['absoluteCurvatureDifference']['count'], 0)
      self.assertTrue(report['truncatedTail'])
      self.assertEqual(report['failureCount'], 1)

  def test_corrupt_middle_line_is_not_silently_ignored(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'live.jsonl'
      path.write_text('bad-json\n{}\n')
      with self.assertRaises(json.JSONDecodeError):
        summarize_log(path)


if __name__ == '__main__':
  unittest.main()
