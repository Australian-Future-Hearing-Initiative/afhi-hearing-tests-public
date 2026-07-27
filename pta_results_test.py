'''Unit tests for CSV generators in pta_results.py.'''

import pta_results


# Sample trial data: (ear, frequency, dbhl, heard, response_time_s).
_SAMPLE_TRIALS = [
    ('left', 1000, 40, True, 1.1234567),
    ('right', 2000, 50, False, 0.9876543),
]

_SAMPLE_AUDIOGRAM = [(1000, 30), (2000, 40), (4000, 50)]


def test_generate_pta_full_results_csv_headers():
  '''CSV contains all expected comment header lines.'''
  csv = pta_results.generate_pta_full_results_csv(
      _SAMPLE_TRIALS, pta_duration_s=120, pta_method='Hybrid'
  )
  assert '# Pure-Tone Audiometry Test Results' in csv
  assert '# Test Method: Hybrid' in csv
  assert '# Test date/time (UTC):' in csv
  assert f'# Tones presented: {len(_SAMPLE_TRIALS)}' in csv
  assert '# Test duration: 120 s' in csv
  assert '# System volume:' in csv


def test_generate_pta_full_results_csv_pause_headers():
  '''CSV contains active duration, pause count, and paused duration headers.'''
  csv = pta_results.generate_pta_full_results_csv(
      _SAMPLE_TRIALS, pta_duration_s=120, pta_method='Hybrid',
      active_duration_s=105.5, pause_count=2, total_pause_duration_s=14.5
  )
  assert '# Test duration: 120 s' in csv
  assert '# Active duration: 105 s' in csv
  assert '# Pause count: 2' in csv
  assert '# Total paused duration: 14 s' in csv


def test_generate_pta_full_results_csv_omitted_pause_args():
  '''Omitting optional pause args does not add pause header lines.'''
  csv = pta_results.generate_pta_full_results_csv(
      _SAMPLE_TRIALS, pta_duration_s=120, pta_method='Hybrid'
  )
  assert '# Active duration:' not in csv
  assert '# Pause count:' not in csv
  assert '# Total paused duration:' not in csv


def test_generate_pta_full_results_csv_row_count():
  '''CSV data section contains exactly one row per trial (plus header row).'''
  csv = pta_results.generate_pta_full_results_csv(
      _SAMPLE_TRIALS, pta_duration_s=60, pta_method='Hybrid'
  )
  # Strip comment lines.
  data_lines = [l for l in csv.splitlines() if not l.startswith('#')]
  # One header row + one row per trial.
  assert len(data_lines) == len(_SAMPLE_TRIALS) + 1


def test_generate_pta_full_results_csv_response_time_rounded():
  '''Response times are rounded to 3 decimal places in the output.'''
  csv = pta_results.generate_pta_full_results_csv(
      _SAMPLE_TRIALS, pta_duration_s=60, pta_method='Hybrid'
  )
  # 1.1234567 should appear as 1.123.
  assert '1.123' in csv
  # Full unrounded value must not appear.
  assert '1.1234567' not in csv


def test_generate_audiogram_csv_headers():
  '''Audiogram CSV contains ear label and frequency/threshold column header.'''
  csv = pta_results.generate_audiogram_csv('Left', _SAMPLE_AUDIOGRAM)
  assert '# Left-Ear Audiogram' in csv
  assert '# Frequency (Hz), Threshold (dB HL)' in csv


def test_generate_audiogram_csv_data_rows():
  '''Audiogram CSV contains one row per (frequency, threshold) pair.'''
  csv = pta_results.generate_audiogram_csv('Right', _SAMPLE_AUDIOGRAM)
  data_lines = [l for l in csv.splitlines() if not l.startswith('#')]
  assert len(data_lines) == len(_SAMPLE_AUDIOGRAM)
  assert '1000,30' in csv
  assert '4000,50' in csv
