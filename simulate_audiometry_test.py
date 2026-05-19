'''Unit tests for pure-function helpers in simulate_audiometry.py.'''

# pylint: disable=protected-access

import numpy as np
import pytest

from simulate_audiometry import (
  _calculate_mae,
  determine_hybrid_phase,
  get_interpolated_threshold,
)
from common import STANDARD_FREQS_HZ


# ---------------------------------------------------------------------------
# get_interpolated_threshold
# ---------------------------------------------------------------------------

def test_get_interpolated_threshold_exact_knot():
  '''Returns exact value when frequency matches an audiogram key.'''
  audiogram = {500: 10.0, 1000: 20.0, 2000: 40.0}
  assert get_interpolated_threshold(1000.0, audiogram) == pytest.approx(20.0)


def test_get_interpolated_threshold_exact_knot_boundary():
  '''Returns exact value for the lowest and highest defined frequency.'''
  audiogram = {250: 5.0, 8000: 60.0}
  assert get_interpolated_threshold(250.0, audiogram) == pytest.approx(5.0)
  assert get_interpolated_threshold(8000.0, audiogram) == pytest.approx(60.0)


def test_get_interpolated_threshold_midpoint_log():
  '''Interpolates at the geometric-mean frequency (midpoint on log2 scale).'''
  audiogram = {1000: 0.0, 4000: 40.0}
  # Geometric mean of 1000 and 4000 is 2000; on a log2 scale that is exactly
  # halfway between log2(1000) and log2(4000), so the result should be 20 dB.
  geo_mean = np.sqrt(1000.0 * 4000.0)  # = 2000.
  result = get_interpolated_threshold(geo_mean, audiogram)
  assert result == pytest.approx(20.0, abs=1e-3)


def test_get_interpolated_threshold_flat_audiogram():
  '''Flat audiogram returns the same value at any frequency.'''
  audiogram = {500: 25.0, 1000: 25.0, 2000: 25.0, 4000: 25.0}
  for freq in [500, 750, 1000, 1500, 2000, 3000, 4000]:
    assert get_interpolated_threshold(float(freq), audiogram) == pytest.approx(
        25.0)


# ---------------------------------------------------------------------------
# _calculate_mae
# ---------------------------------------------------------------------------

def test_calculate_mae_typical():
  '''Typical case: MAE equals mean of absolute differences.'''
  # True audiogram is flat at 20 dB; estimated is off by 5 dB at each freq.
  true_ag = {1000: 20.0, 2000: 20.0, 4000: 20.0}
  estimated = {1000.0: 25.0, 2000.0: 15.0, 4000.0: 25.0}
  result = _calculate_mae(estimated, true_ag)
  assert result == pytest.approx(5.0)


def test_calculate_mae_perfect():
  '''Zero error when estimates exactly match the true audiogram.'''
  true_ag = {1000: 30.0, 2000: 30.0}
  estimated = {1000.0: 30.0, 2000.0: 30.0}
  assert _calculate_mae(estimated, true_ag) == pytest.approx(0.0)


def test_calculate_mae_all_none():
  '''Returns None when all estimated thresholds are None.'''
  true_ag = {1000: 20.0, 2000: 30.0}
  estimated = {1000.0: None, 2000.0: None}
  assert _calculate_mae(estimated, true_ag) is None


def test_calculate_mae_empty_estimated():
  '''Returns None for an empty estimated audiogram.'''
  assert _calculate_mae({}, {1000: 20.0}) is None


# ---------------------------------------------------------------------------
# determine_hybrid_phase
# ---------------------------------------------------------------------------

# Target frequency used by the hybrid selector.
_TARGET_FREQ = 1000.0
_MIN_LEVEL = -5.0  # PTA_MIN_LEVEL_DB_HL from common.py.
# All standard frequencies other than the target.
_OTHER_FREQS = [f for f in STANDARD_FREQS_HZ if f != _TARGET_FREQ]


def test_determine_hybrid_phase_descent_empty():
  '''Empty history is always in Descent (Phase 1).'''
  assert determine_hybrid_phase([]) == 'Descent'


def test_determine_hybrid_phase_descent_heard_once():
  '''Hearing the target once (not at min level) stays in Descent.'''
  history = [(_TARGET_FREQ, 60.0, True)]
  assert determine_hybrid_phase(history) == 'Descent'


def test_determine_hybrid_phase_sweep_after_not_heard():
  '''Finding a "not heard" at the target freq transitions to Sweep.'''
  history = [
      (_TARGET_FREQ, 60.0, True),
      (_TARGET_FREQ, 50.0, False),
  ]
  assert determine_hybrid_phase(history) == 'Sweep'


def test_determine_hybrid_phase_sweep_after_min_level_heard():
  '''Hearing the target at min level also ends Descent and enters Sweep.'''
  history = [(_TARGET_FREQ, _MIN_LEVEL, True)]
  assert determine_hybrid_phase(history) == 'Sweep'


def test_determine_hybrid_phase_sweep_mid_sweep():
  '''Still Sweep when only some non-target frequencies have been tested.'''
  history = [
      (_TARGET_FREQ, 60.0, True),
      (_TARGET_FREQ, 50.0, False),
      (_OTHER_FREQS[0], 55.0, True),  # Only one non-target tested.
  ]
  assert determine_hybrid_phase(history) == 'Sweep'


def test_determine_hybrid_phase_adaptive():
  '''All non-target frequencies tested transitions to Adaptive (Phase 3).'''
  history = [
      (_TARGET_FREQ, 60.0, True),
      (_TARGET_FREQ, 50.0, False),
  ] + [(f, 60.0, True) for f in _OTHER_FREQS]
  assert determine_hybrid_phase(history) == 'Adaptive'
