'''Unit tests for cls_audiogram.py.'''

import numpy as np
import pytest

import calibration
import cls_audiogram
import common
import pta_results

_LEFT = [(250, 10), (500, 15), (1000, 20), (2000, 40), (4000, 60),
         (8000, 70)]
_RIGHT = [(250, 5), (500, 5), (1000, 10), (2000, 20), (4000, 30),
          (8000, 35)]
_REF = cls_audiogram.FLOOR_REFERENCE_CONFIGURATION


def _pta_zip_bytes():
  '''Builds a zip shaped like the PTA test's results download.'''
  return common.generate_zip_bytes([
      ('pta_full_results.csv', 'ignored'),
      ('pta_left_audiogram.csv',
       pta_results.generate_audiogram_csv('Left', _LEFT)),
      ('pta_right_audiogram.csv',
       pta_results.generate_audiogram_csv('Right', _RIGHT)),
  ])


def test_parse_round_trips_pta_export():
  '''Parses the CSV exactly as the PTA test writes it.'''
  csv = pta_results.generate_audiogram_csv('Right', _RIGHT[::-1])
  ear, freqs, hls = cls_audiogram.parse_audiogram_csv(csv, 'x.csv')
  assert ear == 'right'
  np.testing.assert_array_equal(freqs, [f for f, _ in _RIGHT])
  np.testing.assert_array_equal(hls, [h for _, h in _RIGHT])


def test_parse_falls_back_to_filename_for_ear():
  '''Uses the file name when the ear header is missing.'''
  ear, _, _ = cls_audiogram.parse_audiogram_csv('1000,20\n', 'left.csv')
  assert ear == 'left'


def test_parse_drops_non_finite_thresholds():
  '''Skips rows without a usable threshold.'''
  csv = '# Left-Ear Audiogram\n1000,20\n2000,nan\n4000,\n'
  _, freqs, hls = cls_audiogram.parse_audiogram_csv(csv)
  np.testing.assert_array_equal(freqs, [1000])
  np.testing.assert_array_equal(hls, [20])


@pytest.mark.parametrize('csv, name', [
    ('1000,20\n', 'audiogram.csv'),  # Ear unknown.
    ('# Left-Ear Audiogram\n', 'a.csv'),  # No data.
    ('# Left-Ear Audiogram\nfoo,bar\n', 'a.csv'),  # Not numeric.
    ('# Left-Ear Audiogram\n1000,20,5\n', 'a.csv'),  # Too many columns.
    ('left,1000,20,True\n', 'pta_left_x.csv'),  # Wrong file, ear in name.
])
def test_parse_rejects_bad_input(csv, name):
  '''Raises ValueError for unusable files.'''
  with pytest.raises(ValueError):
    cls_audiogram.parse_audiogram_csv(csv, name)


def test_load_from_pta_zip():
  '''Loads both ears from the PTA results zip and ignores other files.'''
  audiogram = cls_audiogram.load_audiogram_files(
      [('UTC20261104_pta_results.zip', _pta_zip_bytes())])
  assert set(audiogram) == {'left', 'right'}
  np.testing.assert_array_equal(audiogram['left'][1], [h for _, h in _LEFT])


def test_load_from_single_csv():
  '''A single ear can be supplied on its own.'''
  csv = pta_results.generate_audiogram_csv('Left', _LEFT).encode()
  audiogram = cls_audiogram.load_audiogram_files(
      [('pta_left_audiogram.csv', csv)])
  assert set(audiogram) == {'left'}


def test_load_rejects_duplicate_ear():
  '''The same ear cannot be supplied twice.'''
  csv = pta_results.generate_audiogram_csv('Left', _LEFT).encode()
  with pytest.raises(ValueError, match='left'):
    cls_audiogram.load_audiogram_files([('a.csv', csv), ('b.csv', csv)])


def test_load_rejects_zip_without_audiogram():
  '''A zip with no audiogram CSVs is an error.'''
  data = common.generate_zip_bytes([('other.csv', 'x')])
  with pytest.raises(ValueError):
    cls_audiogram.load_audiogram_files([('r.zip', data)])


def test_load_ignores_macos_zip_metadata():
  '''A zip re-compressed in Finder (subfolder + __MACOSX entries) loads.'''
  left = pta_results.generate_audiogram_csv('Left', _LEFT)
  data = common.generate_zip_bytes([
      ('results/pta_left_audiogram.csv', left),
      ('__MACOSX/results/._pta_left_audiogram.csv', b'\x00\x05\x16\x07\xff'),
      ('results/.hidden_audiogram.csv', b'\xff\xfe'),
  ])
  audiogram = cls_audiogram.load_audiogram_files([('r.zip', data)])
  assert set(audiogram) == {'left'}


def test_load_rejects_binary_csv():
  '''A non-text file gives a clear error rather than a decode failure.'''
  with pytest.raises(ValueError, match='not a text CSV'):
    cls_audiogram.load_audiogram_files([('left.csv', b'\xff\xfe\x00\x81')])


def test_load_rejects_corrupt_zip():
  '''A file named .zip that is not a zip is an error.'''
  with pytest.raises(ValueError):
    cls_audiogram.load_audiogram_files([('r.zip', b'not a zip')])


def test_threshold_at_audiogram_frequency_matches_calibration():
  '''At a tested frequency, the threshold is the calibration conversion.'''
  freqs, hls = np.array([1000., 4000.]), np.array([20., 60.])
  assert cls_audiogram.threshold_dbspl(4000, freqs, hls) == pytest.approx(
      calibration.dbhl_to_dbspl(60, 4000, _REF))


def test_threshold_interpolates_in_log_frequency():
  '''Midway in log-frequency gives the mean of the end points.'''
  freqs, hls = np.array([1000., 4000.]), np.array([20., 60.])
  mid = cls_audiogram.threshold_dbspl(2000, freqs, hls)
  ref = calibration.dbhl_to_dbspl(0, 2000, _REF)
  assert mid == pytest.approx(40 + ref)


def test_threshold_held_constant_beyond_audiogram():
  '''Outside the tested range the nearest threshold is used.'''
  freqs, hls = np.array([1000., 4000.]), np.array([20., 60.])
  low = cls_audiogram.threshold_dbspl(250, freqs, hls)
  assert low == pytest.approx(calibration.dbhl_to_dbspl(20, 250, _REF))


def test_floor_adds_margin_and_uses_better_ear_for_both():
  '''Binaural floor is the better ear's threshold plus the margin.'''
  audiogram = cls_audiogram.load_audiogram_files(
      [('r.zip', _pta_zip_bytes())])
  margin = cls_audiogram.AUDIOGRAM_MARGIN_DB
  left = cls_audiogram.floor_dbspl(4000, 'left', audiogram)
  right = cls_audiogram.floor_dbspl(4000, 'right', audiogram)
  assert left == pytest.approx(
      calibration.dbhl_to_dbspl(60, 4000, _REF) + margin)
  assert cls_audiogram.floor_dbspl(4000, 'both', audiogram) == min(left,
                                                                   right)


def test_floor_is_none_for_missing_ear():
  '''No floor applies to an ear without an audiogram.'''
  audiogram = {'left': (np.array([1000.]), np.array([20.]))}
  assert cls_audiogram.floor_dbspl(1000, 'right', audiogram) is None
  assert cls_audiogram.floor_amplitude(1000, 'right', audiogram) is None
  assert cls_audiogram.floor_dbspl(1000, 'both', audiogram) is not None


def _draws(candidates):
  '''Returns a draw function yielding the given candidates in order.'''
  it = iter(candidates)
  return lambda: next(it)


def test_draw_skips_candidates_below_floor():
  '''Candidates below the floor are redrawn.'''
  draw = _draws([(1000, 0.1), (1000, 0.4), (1000, 0.6)])
  result = cls_audiogram.draw_audible_stimulus(draw, lambda f: 0.5, 1.0)
  assert result == (1000, 0.6)


def test_draw_skips_frequencies_with_floor_above_cap():
  '''A frequency whose floor exceeds the volume cap is never played.'''
  floors = {8000: 0.9, 1000: 0.2}
  draw = _draws([(8000, 1.0), (1000, 0.3)])
  result = cls_audiogram.draw_audible_stimulus(draw, floors.get, 0.5)
  assert result == (1000, 0.3)


def test_draw_accepts_anything_without_floor():
  '''No floor means the first candidate is used.'''
  draw = _draws([(1000, 0.001)])
  assert cls_audiogram.draw_audible_stimulus(
      draw, lambda f: None, 1.0) == (1000, 0.001)


def test_draw_falls_back_after_max_attempts():
  '''If nothing is audible, the last candidate is returned.'''
  draw = _draws([(1000, 0.1), (2000, 0.2), (3000, 0.3)])
  result = cls_audiogram.draw_audible_stimulus(
      draw, lambda f: 0.9, 1.0, max_attempts=3)
  assert result == (3000, 0.3)


def test_generate_floor_csv():
  '''The results CSV records settings and one row per ear and frequency.'''
  audiogram = cls_audiogram.load_audiogram_files(
      [('r.zip', _pta_zip_bytes())])
  csv = cls_audiogram.generate_floor_csv(audiogram)
  assert f'# Margin above threshold: {cls_audiogram.AUDIOGRAM_MARGIN_DB}' in csv
  assert f'# dB HL to dB SPL reference: {_REF}' in csv
  assert 'Ear,Frequency (Hz),Threshold (dB HL),Floor (dB SPL)' in csv
  data_rows = [r for r in csv.splitlines()[4:] if r]
  assert len(data_rows) == len(_LEFT) + len(_RIGHT)
