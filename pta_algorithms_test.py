"""Unit tests for pta_algorithms.py."""

# pylint: disable=protected-access

import numpy as np
import pytest
from unittest.mock import patch

from pta_algorithms import (
    KernelSelector,
    HybridSelector,
    KernelReconstructor,
    HybridLogisticReconstructor,
)
import adaptive_audiometry
from common import STANDARD_FREQS_HZ


def test_kernel_selector_init():
  """Tests KernelSelector copies the default config."""
  selector = KernelSelector()
  assert selector.kernel_config == adaptive_audiometry.KERNEL_SMOOTHER_CONFIG
  assert (
      selector.kernel_config is not adaptive_audiometry.KERNEL_SMOOTHER_CONFIG
  )


@patch('pta_algorithms.adaptive_audiometry.run_adaptive_test_step')
def test_kernel_selector_next_stimulus(mock_run_adaptive):
  """Tests KernelSelector correctly routes data to the adaptive module."""
  mock_run_adaptive.return_value = ((1000.0, 50.0), {'mock': 'audiogram'})
  selector = KernelSelector()
  history = [(1000.0, 60.0, True)]

  next_freq, next_level = selector.next_stimulus(history, verbosity=3)

  assert next_freq == 1000.0
  assert next_level == 50.0
  mock_run_adaptive.assert_called_once()
  # Verify history was passed correctly
  _, kwargs = mock_run_adaptive.call_args
  assert kwargs['past_results'] == history
  assert kwargs['verbosity'] == 3


def test_kernel_reconstructor_init():
  """Tests KernelReconstructor expands the grid bounds."""
  reconstructor = KernelReconstructor()
  # It should subtract 10 from min and add 10 to max
  assert (
      reconstructor.grid_config['min_level_dbhl']
      == adaptive_audiometry.GRID_CONFIG['min_level_dbhl'] - 10
  )
  assert (
      reconstructor.grid_config['max_level_dbhl']
      == adaptive_audiometry.GRID_CONFIG['max_level_dbhl'] + 10
  )


def test_kernel_reconstructor_empty_history():
  """Tests KernelReconstructor returns Nones for empty history."""
  reconstructor = KernelReconstructor()
  result = reconstructor.reconstruct([], verbosity=3)

  assert 'Kernel' in result
  assert reconstructor.final_kernel_probs is None
  # All standard freqs should map to None
  for freq in STANDARD_FREQS_HZ:
    assert result['Kernel'][freq] is None


@patch('pta_algorithms.adaptive_audiometry.fit_kernel_smoother')
@patch('pta_algorithms.adaptive_audiometry.estimate_thresholds')
def test_kernel_reconstructor_active_history(mock_estimate, mock_fit):
  """Tests KernelReconstructor strings the adaptive functions together."""
  mock_fit.return_value = np.array([0.5, 0.5])
  mock_estimate.return_value = {1000.0: 30.0}

  reconstructor = KernelReconstructor()
  history = [(1000.0, 60.0, True)]
  result = reconstructor.reconstruct(history, verbosity=3)

  assert 'Kernel' in result
  assert result['Kernel'] == {1000.0: 30.0}
  assert reconstructor.final_kernel_probs is not None
  mock_fit.assert_called_once()
  mock_estimate.assert_called_once()


def test_hybrid_selector_init_max_level():
  """Tests HybridSelector allows max level override."""
  selector = HybridSelector(max_level_dbhl=85.0)
  assert selector.initial_phase_config['max_level_dbhl'] == 85.0


@patch.object(HybridLogisticReconstructor, 'reconstruct')
def test_hybrid_selector_phase2_fallback(mock_reconstruct):
  """Tests Phase 2 falls back to 1kHz threshold if global fit fails."""
  selector = HybridSelector()
  # Provide history indicating 1000 Hz threshold found (Phase 1 complete)
  history = [(1000.0, 60.0, True), (1000.0, 50.0, False)]

  # Mock global fit missing the next sweep frequency (e.g., 2000.0)
  mock_reconstruct.return_value = {'Global Parametric': {}}

  next_freq, next_level = selector.next_stimulus(history, verbosity=3)

  # It should pick an untested freq (2000) but fall back to the 1000 Hz
  # threshold (60 dB) instead of crashing.
  assert next_freq != 1000.0
  assert next_level == 60.0


@patch.object(HybridLogisticReconstructor, 'reconstruct')
def test_hybrid_selector_phase3_no_variances(mock_reconstruct):
  """Tests Phase 3 raises ValueError if all variances are None."""
  selector = HybridSelector()
  # Provide history hitting all standard freqs (Phase 3 active)
  history = [(f, 60.0, True) for f in STANDARD_FREQS_HZ]
  # Add one 'False' so Phase 1 logic doesn't think 1kHz isn't fully tested
  history.append((1000.0, 50.0, False))

  # Mock reconstruct to return None for variances
  variances = {f: None for f in STANDARD_FREQS_HZ}
  mock_reconstruct.return_value = {
      'Hybrid': {'variances': variances, 'thresholds': {}}
  }

  with pytest.raises(ValueError, match='No valid variances found'):
    selector.next_stimulus(history, verbosity=3)


@patch.object(HybridLogisticReconstructor, 'reconstruct')
def test_hybrid_selector_phase3_missing_threshold(mock_reconstruct):
  """Tests Phase 3 raises ValueError if max variance freq has no threshold."""
  selector = HybridSelector()
  # Provide history hitting all standard freqs (Phase 3 active)
  history = [(f, 60.0, True) for f in STANDARD_FREQS_HZ]
  history.append((1000.0, 50.0, False))

  # Provide a valid variance for 2000, but no threshold
  mock_reconstruct.return_value = {
      'Hybrid': {'variances': {2000.0: 10.0}, 'thresholds': {2000.0: None}}
  }

  with pytest.raises(
      ValueError, match='Threshold for frequency 2000.0 Hz is None'
  ):
    selector.next_stimulus(history, verbosity=3)


def test_hybrid_selector_phase1():
  """Tests Phase 1 initial descent logic in HybridSelector."""
  selector = HybridSelector()
  history = [(1000.0, 60.0, True)]
  next_freq, next_level = selector.next_stimulus(history, verbosity=3)
  assert next_freq == 1000.0
  assert next_level == 50.0


@patch.object(HybridLogisticReconstructor, 'reconstruct')
def test_hybrid_selector_phase3_success(mock_reconstruct):
  """Tests Phase 3 correctly selects max variance freq and threshold."""
  selector = HybridSelector()
  history = [(f, 60.0, True) for f in STANDARD_FREQS_HZ]
  history.append((1000.0, 50.0, False))

  mock_reconstruct.return_value = {
      'Hybrid': {
          'variances': {2000.0: 10.0, 4000.0: 5.0},
          'thresholds': {2000.0: 30.0, 4000.0: 20.0},
      }
  }

  next_freq, next_level = selector.next_stimulus(history, verbosity=3)

  assert next_freq == 2000.0
  assert next_level == 30.0
