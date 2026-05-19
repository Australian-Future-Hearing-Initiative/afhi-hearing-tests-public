"""Integration test for adaptive PTA algorithm portability.

Verifies that the algorithm produces correct final thresholds given a
ground-truth audiogram. This test is designed to be easily portable to
other platforms (e.g., Android/Kotlin).

The golden JSON files serve as the contract:
  Input:  ground_truth_audiogram + num_trials
  Output: final_estimation.thresholds

Response model: heard = (stimulus_level_dbhl > true_threshold_at_freq)
"""

import json
import os
import unittest
from glob import glob

from pta_algorithms import HybridSelector, HybridLogisticReconstructor
from simulate_audiometry import get_interpolated_threshold


class TestAdaptivePTAPortability(unittest.TestCase):
  """Verifies adaptive PTA output against golden data."""

  GOLDEN_DATA_DIR = 'tests/golden_data'
  # Allow minor cross-platform float differences. 0.5 dB is clinically
  # insignificant but gives headroom for numerical differences.
  TOLERANCE_DB = 0.5

  def test_all_scenarios(self):
    """Run each scenario and verify final thresholds."""
    json_files = glob(os.path.join(self.GOLDEN_DATA_DIR, '*.json'))
    self.assertTrue(json_files,
                    f'No golden data found in {self.GOLDEN_DATA_DIR}. '
                    'Run generate_golden_data.py first.')

    for json_path in json_files:
      with self.subTest(scenario=os.path.basename(json_path)):
        self._verify_scenario(json_path)

  def _verify_scenario(self, json_path: str):
    """Re-simulates a scenario and compares final thresholds."""
    with open(json_path, 'r', encoding='utf-8') as f:
      data = json.load(f)

    ground_truth = {float(k): v
                    for k, v in data['ground_truth_audiogram'].items()}
    num_trials = data['simulation_parameters']['num_trials']
    expected = {float(k): v
                for k, v in data['final_estimation']['thresholds'].items()}

    # Run full simulation.
    selector = HybridSelector()
    reconstructor = HybridLogisticReconstructor()
    history = []

    for _ in range(num_trials):
      freq, level = selector.next_stimulus(history, verbosity=0)
      threshold = get_interpolated_threshold(freq, ground_truth)
      # Deterministic response model: heard if level is above threshold.
      response = level > threshold
      history.append((freq, level, response))

    # Get final thresholds.
    result = reconstructor.reconstruct(history, verbosity=0)
    actual = result.get('Hybrid', {}).get('thresholds', {})

    # Compare frequencies.
    self.assertEqual(
        set(actual.keys()), set(expected.keys()),
        msg=f'Frequency set mismatch in {os.path.basename(json_path)}'
    )

    # Compare threshold values.
    for freq in expected:
      self.assertAlmostEqual(
          actual[freq], expected[freq], delta=self.TOLERANCE_DB,
          msg=f'Threshold mismatch at {freq} Hz'
      )


if __name__ == '__main__':
  unittest.main()
