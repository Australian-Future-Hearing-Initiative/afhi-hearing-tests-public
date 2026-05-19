"""Tests for audiogram.py."""

import unittest
from parameterized import parameterized

import audiogram

# Constants used in tests, matching those in demo_pta.py
MIN_LEVEL_DB_HL = -5
MAX_LEVEL_DB_HL = 70

class TestAudiogramFitting(unittest.TestCase):
  """Tests for the audiogram fitting functions."""

  @parameterized.expand([
    ('no_data',
     []),
    ('not_enough_data',
     [(40, True)]),
    ('min_level_but_only_twice',
     [(MIN_LEVEL_DB_HL, True), (MIN_LEVEL_DB_HL, True)]),
    ('descending_only_always_heard',
     [(60, True), (50, True), (40, True), (30, True)]),
    ('only_one_ascending_level',
     [(50, True), (40, False), (45, True), (45, True)]),
    ('descending_only_no_threshold',
     [(50, True), (40, True), (30, True), (20, False)]),
    ('two_different_qualifying_levels',
     [(40, False), (45, True), (40, False), (30, False), (35, True)]),
  ])
  def test_get_threshold_returns_none(self, test_case_name, results):
    """Test get_threshold with various scenarios that should result in None."""
    actual_result = audiogram.get_threshold(results, MIN_LEVEL_DB_HL,
                                            MAX_LEVEL_DB_HL)
    self.assertEqual(actual_result, None,
                     f"Test '{test_case_name}' Failed: '"
                     f'Expected {None}, got {actual_result}')

  @parameterized.expand([
    ('two_consistent_levels',
     [(50, True), (40, False), (45, True), (40, False), (45, True)], 45),
    ('three_levels_take_median',
     [(50, True), (40, False), (45, True), (35, False), (40, True), (30, False),
      (35, True)], 40),
    ('min_level_three_times',
     [(MIN_LEVEL_DB_HL, True), (MIN_LEVEL_DB_HL, True),
      (MIN_LEVEL_DB_HL, True)], MIN_LEVEL_DB_HL),
    ('max_level_not_heard',
      [(MAX_LEVEL_DB_HL, False), (MAX_LEVEL_DB_HL, False)],
      MAX_LEVEL_DB_HL),
  ])
  def test_get_threshold_returns_correct_result(self, test_case_name, results,
                                                expected_result):
    """Test get_threshold with various valid scenarios."""
    actual_result = audiogram.get_threshold(results, MIN_LEVEL_DB_HL,
                                            MAX_LEVEL_DB_HL)
    self.assertEqual(actual_result, expected_result,
                     f"Test '{test_case_name}' Failed: '"
                     f'Expected {expected_result}, got {actual_result}')

if __name__ == '__main__':
  unittest.main()
