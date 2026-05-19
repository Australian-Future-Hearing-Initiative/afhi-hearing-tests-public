'''Unit tests for CSV generators in pip_results.py.'''

import pip_results


# Sample trial data: (ear, frequency, dbhl, heard, response_time_s).
_SAMPLE_TRIALS = [
    ('left', 1000, 40, True, 1.1234567),
    ('right', 2000, 50, False, 0.9876543),
]

_SAMPLE_AUDIOGRAM = [(1000, 25), (2000, 35), (4000, 45)]


def test_generate_pip_full_results_csv_headers():
  '''CSV contains all expected comment header lines.'''
  csv = pip_results.generate_pip_full_results_csv(
      _SAMPLE_TRIALS, pip_duration_s=90
  )
  assert '# Pip-Based Pure-Tone Audiometry Test Results' in csv
  assert '# Test date/time (UTC):' in csv
  assert f'# Tones presented: {len(_SAMPLE_TRIALS)}' in csv
  assert '# Test duration: 90 s' in csv
  assert '# System volume:' in csv


def test_generate_pip_full_results_csv_row_count():
  '''CSV data section contains exactly one row per trial (plus header row).'''
  csv = pip_results.generate_pip_full_results_csv(
      _SAMPLE_TRIALS, pip_duration_s=90
  )
  data_lines = [l for l in csv.splitlines() if not l.startswith('#')]
  assert len(data_lines) == len(_SAMPLE_TRIALS) + 1


def test_generate_pip_full_results_csv_response_time_rounded():
  '''Response times are rounded to 3 decimal places in the output.'''
  csv = pip_results.generate_pip_full_results_csv(
      _SAMPLE_TRIALS, pip_duration_s=90
  )
  assert '1.123' in csv
  assert '1.1234567' not in csv


def test_generate_pip_audiogram_csv_headers():
  '''Audiogram CSV contains ear label and column header.'''
  csv = pip_results.generate_pip_audiogram_csv('Left', _SAMPLE_AUDIOGRAM)
  assert '# Left-Ear Audiogram' in csv
  assert '# Frequency (Hz), Threshold (dB HL)' in csv


def test_generate_pip_audiogram_csv_data_rows():
  '''Audiogram CSV contains one row per (frequency, threshold) pair.'''
  csv = pip_results.generate_pip_audiogram_csv('Right', _SAMPLE_AUDIOGRAM)
  data_lines = [l for l in csv.splitlines() if not l.startswith('#')]
  assert len(data_lines) == len(_SAMPLE_AUDIOGRAM)
  assert '1000,25' in csv
  assert '4000,45' in csv
