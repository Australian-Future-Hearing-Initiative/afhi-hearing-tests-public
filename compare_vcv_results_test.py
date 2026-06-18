"""Unit tests for compare_vcv_results.py."""

import os

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import compare_vcv_results as cvr


def _make_standard_estimates_df(ear: str = 'both') -> pd.DataFrame:
  """Builds a minimal valid estimates DataFrame for the standard 10."""
  srt_by_consonant = {
      'B': -6.0, 'V': -2.0,
      'Z': -14.0, 'T': -14.5, 'S': -13.5, 'SH': -15.0,
      'N': -10.0, 'D': -3.0, 'K': -11.0, 'G': 1.0,
  }
  rows = []
  for consonant, srt in srt_by_consonant.items():
    rows.append({
        'Ear': ear,
        'Consonant': consonant,
        'SRT (dB)': srt,
        'Uncertainty (SD)': 1.5,
        'Trials': 10,
    })
  return pd.DataFrame(rows)


def test_get_required_members_standard_battery():
  """Standard 10-consonant files require fixed class subsets."""
  required_c1 = cvr.get_required_members('C1', cvr.STANDARD_10)
  required_c2 = cvr.get_required_members('C2', cvr.STANDARD_10)
  required_c3 = cvr.get_required_members('C3', cvr.STANDARD_10)
  assert required_c1 == {'B', 'V'}
  assert required_c2 == {'Z', 'T', 'S', 'SH'}
  assert required_c3 == {'N', 'D', 'K', 'G'}


def test_get_ear_mode_both_and_separate():
  """Ear mode detection accepts binaural and separate-ear data."""
  both_df = _make_standard_estimates_df(ear='both')
  separate_df = pd.concat([
      _make_standard_estimates_df(ear='left'),
      _make_standard_estimates_df(ear='right'),
  ], ignore_index=True)
  assert cvr.get_ear_mode(both_df) == 'both'
  assert cvr.get_ear_mode(separate_df) == 'separate'


def test_compute_class_statistics_known_means():
  """Class means match the simple average of member consonants."""
  df = _make_standard_estimates_df()
  stats = cvr.compute_class_statistics(df, ear='both', condition_name='test')
  assert stats['C1']['mean'] == pytest.approx(-4.0)
  assert stats['C2']['mean'] == pytest.approx(-14.25)
  assert stats['C3']['mean'] == pytest.approx(-5.75)
  assert stats['C1']['ci_half'] > 0


def test_create_comparison_plot_returns_figure():
  """Plot creation succeeds for valid class statistics."""
  stats = cvr.compute_class_statistics(
      _make_standard_estimates_df(), ear='both', condition_name='a',
  )
  fig = cvr.create_comparison_plot({'condition_a': stats})
  assert isinstance(fig, plt.Figure)
  plt.close(fig)


def test_run_comparison_writes_png(tmp_path):
  """End-to-end run saves a PNG for a valid comparison folder."""
  for name in ('cond_a', 'cond_b'):
    out_dir = tmp_path / name
    out_dir.mkdir()
    _make_standard_estimates_df().to_csv(
        out_dir / cvr.ESTIMATES_FILENAME, index=False,
    )

  output_dir = tmp_path / 'plots'
  original_output_dir = cvr.OUTPUT_DIR
  cvr.OUTPUT_DIR = str(output_dir)
  try:
    paths = cvr.run_comparison(str(tmp_path))
  finally:
    cvr.OUTPUT_DIR = original_output_dir

  assert len(paths) == 1
  assert os.path.isfile(paths[0])
  assert paths[0].startswith(str(output_dir))


def test_discover_condition_dirs_fails_on_missing_csv(tmp_path):
  """Missing estimates files abort the run."""
  (tmp_path / 'empty_condition').mkdir()
  with pytest.raises(SystemExit):
    cvr.discover_condition_dirs(str(tmp_path))
