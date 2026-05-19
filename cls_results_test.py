'''Unit tests for the CSV generator in cls_results.py.'''

from unittest.mock import patch

import pandas as pd

import cls_results


_MOCK_VOLUME = '50'

_VOLUME_PATCH = 'cls_results.common.get_macos_system_volume'


def _make_df(n_rows=3):
  '''Returns a minimal CLS results DataFrame.'''
  return pd.DataFrame({
      'Level (dB SPL)': [40, 50, 60][:n_rows],
      'Response': ['soft', 'medium', 'loud'][:n_rows],
      'Response Time (s)': [1.1234567, 0.9876543, 2.3456789][:n_rows],
  })


def test_generate_cls_results_csv_headers():
  '''CSV contains all expected comment header lines.'''
  with patch(_VOLUME_PATCH, return_value=_MOCK_VOLUME):
    csv = cls_results.generate_cls_results_csv(_make_df(), duration_s=180)
  assert '# Categorical Loudness Scaling Test Results' in csv
  assert '# Test date/time (UTC):' in csv
  assert '# Number of stimuli: 3' in csv
  assert '# Test duration: 180 s' in csv
  assert f'# System volume: {_MOCK_VOLUME}' in csv


def test_generate_cls_results_csv_row_count():
  '''CSV data section contains exactly one row per DataFrame row.'''
  df = _make_df(n_rows=3)
  with patch(_VOLUME_PATCH, return_value=_MOCK_VOLUME):
    csv = cls_results.generate_cls_results_csv(df, duration_s=60)
  data_lines = [l for l in csv.splitlines() if not l.startswith('#')]
  # One header row + one row per df row.
  assert len(data_lines) == len(df) + 1


def test_generate_cls_results_csv_response_time_rounded():
  '''Response times are rounded to 3 decimal places.'''
  with patch(_VOLUME_PATCH, return_value=_MOCK_VOLUME):
    csv = cls_results.generate_cls_results_csv(_make_df(), duration_s=60)
  assert '1.123' in csv
  assert '1.1234567' not in csv


def test_generate_cls_results_csv_no_response_time_column():
  '''Function handles a DataFrame without a Response Time column gracefully.'''
  df = pd.DataFrame({'Level (dB SPL)': [40, 50], 'Response': ['soft', 'loud']})
  with patch(_VOLUME_PATCH, return_value=_MOCK_VOLUME):
    csv = cls_results.generate_cls_results_csv(df, duration_s=30)
  assert '# Categorical Loudness Scaling Test Results' in csv
  assert '# Number of stimuli: 2' in csv
