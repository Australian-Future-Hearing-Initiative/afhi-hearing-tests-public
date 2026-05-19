""" Defines the abstract base classes and concrete implementations for Pure-Tone
Audiometry (PTA) stimulus selection and audiogram reconstruction algorithms.

This module provides a framework for simulating different PTA strategies. It
separates the logic for choosing the next stimulus (`StimulusSelector`) from the
logic for estimating the final audiogram from the collected data
(`AudiogramReconstructor`).
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional

import numpy as np
import adaptive_audiometry
from logistic_fit import fit_audiogram_hybrid
from common import STANDARD_FREQS_HZ

# Margin (in dB) to add above the global fit threshold during Phase 2 of the
# HybridSelector. This makes it more likely that the tone will be heard, helping
# the UX of the test, while people get used to it.
PHASE2_MARGIN_DB = 5.0

class StimulusSelector(ABC):
  """Abstract base class for a stimulus selection algorithm."""

  @abstractmethod
  def next_stimulus(
      self, history: List[Tuple[float, float, bool]], verbosity: int
  ) -> Tuple[float, float]:
    """
    Determines the next stimulus to present based on the response history.

    Args:
      history: A list of past results, where each result is a tuple of
               (frequency_hz, level_dbhl, response).
      verbosity: The verbosity level for console output.

    Returns:
      A tuple of (frequency_hz, level_dbhl) for the next stimulus.
    """
    pass


class AudiogramReconstructor(ABC):
  """Abstract base class for an audiogram reconstruction algorithm."""

  @abstractmethod
  def reconstruct(
      self, history: List[Tuple[float, float, bool]], verbosity: int
  ) -> Dict[str, any]:
    """
    Reconstructs one or more audiograms from the complete response history.

    Args:
      history: A list of all results from the simulation.
      verbosity: The verbosity level for console output.

    Returns:
      A dictionary where keys are the names of the audiograms
      (e.g., 'Kernel', 'Hybrid') and values are the audiogram data.
    """
    pass


class KernelSelector(StimulusSelector):
  """Selects stimuli using the kernel-based entropy sampling method."""

  def __init__(self):
    """Initializes the selector, setting up configurations."""
    self.kernel_config = adaptive_audiometry.KERNEL_SMOOTHER_CONFIG.copy()

  def next_stimulus(
      self, history: List[Tuple[float, float, bool]], verbosity: int
  ) -> Tuple[float, float]:
    """Determines the next stimulus using the adaptive kernel smoother logic."""
    (next_freq_hz, next_level_dbhl), _ = \
        adaptive_audiometry.run_adaptive_test_step(
        past_results=history,
        standard_freqs=STANDARD_FREQS_HZ,
        kernel_config=self.kernel_config,
        verbosity=verbosity
    )
    return next_freq_hz, next_level_dbhl


class HybridSelector(StimulusSelector):
  """ Selects stimuli using a three-phase hybrid strategy.

  This selector progresses through three distinct phases to efficiently find a
  subject's hearing thresholds:
  1.  **Initial Descent:** Finds an initial threshold at 1 kHz.
  2.  **Broad Sweep:** Tests the other standard frequencies, using the global
      parametric model to choose a smart starting level for each.
  3.  **Adaptive Phase:** After the initial sweep, this phase adaptively
      selects the next stimulus by finding the frequency with the highest
      uncertainty (maximum variance) in the hybrid audiogram fit and testing
      at the current threshold estimate for that frequency.
  """

  def __init__(self, max_level_dbhl=None):
    """Initializes the selector.

    Args:
      max_level_dbhl: Optional override for the maximum stimulus level
        in dB HL. If None, uses the default from INITIAL_PHASE_CONFIG
        (currently 70 dB). Set to 85 for Android-compatible golden data.
    """
    self.reconstructor = HybridLogisticReconstructor()
    # Use the same initial phase config as the adaptive_audiometry module.
    self.initial_phase_config = adaptive_audiometry.INITIAL_PHASE_CONFIG.copy()
    if max_level_dbhl is not None:
      self.initial_phase_config['max_level_dbhl'] = max_level_dbhl

  def next_stimulus(
      self, history: List[Tuple[float, float, bool]], verbosity: int
  ) -> Tuple[float, float]:
    """ Determines the next stimulus based on the current test phase.

    Note: The logic for Phase 1 and Phase 2 is duplicated from
    `adaptive_audiometry.run_adaptive_test_step` to allow for a custom
    adaptive phase (Phase 3). A future refactoring could unify this.
    """
    # --- Phase 1 & 2 Logic (mirrors adaptive_audiometry.run_adaptive_test_step)
    config = self.initial_phase_config
    target_freq = config['target_freq_hz']
    min_level = config['min_level_dbhl']
    max_level = config['max_level_dbhl']

    target_freq_results = [r for r in history if r[0] == target_freq]
    target_freq_not_heard = [r for r in target_freq_results if not r[2]]

    # Phase 1: Initial 1 kHz Descent.
    if not target_freq_not_heard:
      target_freq_heard = [r for r in target_freq_results if r[2]]
      last_heard_level = -np.inf
      if target_freq_heard:
        last_heard_level = min(r[1] for r in target_freq_heard)

      start_or_last_heard = (config['start_level_dbhl']
                             if last_heard_level == -np.inf
                             else last_heard_level)
      next_level = start_or_last_heard - config['descent_step_db']
      next_level = np.clip(next_level, min_level, max_level)
      # Avoid getting stuck at min_level if it has already been heard.
      already_heard_min_level = any(
          r[1] == min_level for r in target_freq_heard)
      if not (next_level == min_level and already_heard_min_level):
        return target_freq, next_level

    # Phase 2: Initial Broad Sweep.
    tested_freqs = {res[0] for res in history if res[0] != target_freq}
    remaining_sweep_freqs = [
        f for f in STANDARD_FREQS_HZ
        if f != target_freq and f not in tested_freqs
    ]
    if remaining_sweep_freqs:
      next_sweep_freq = remaining_sweep_freqs[0]
      if verbosity >= 2:
        print('--- HybridSelector: Phase 2 (Sweep) using Global Fit ---')
      # Use the global parametric fit to estimate the level for the sweep.
      reconstructed = self.reconstructor.reconstruct(history, verbosity)
      global_audiogram = reconstructed.get('Global Parametric', {})
      next_level = global_audiogram.get(next_sweep_freq)
      if next_level is not None:
        next_level += PHASE2_MARGIN_DB

      # Fallback if the global fit doesn't provide a threshold.
      if next_level is None:
        if verbosity >= 1:
          print(f'Warning: Global fit failed for {next_sweep_freq} Hz. '
                f'Falling back to 1kHz threshold.')
        # Start level for sweep is based on 1kHz threshold.
        target_freq_heard = [r for r in target_freq_results if r[2]]
        if not target_freq_heard:
          next_level = config['start_level_dbhl']
        else:
          next_level = min(r[1] for r in target_freq_heard)

      next_level = np.clip(next_level, min_level, max_level)
      return next_sweep_freq, next_level

    # --- Phase 3: Adaptive selection based on logistic fit uncertainty.
    if verbosity >= 2:
      print('--- HybridSelector: Phase 3 (Adaptive) ---')
    reconstructed_audiograms = self.reconstructor.reconstruct(
        history, verbosity)
    hybrid_audiogram = reconstructed_audiograms.get('Hybrid', {})
    variances = hybrid_audiogram.get('variances', {})
    thresholds = hybrid_audiogram.get('thresholds', {})

    # Find the frequency with the maximum variance (highest uncertainty).
    valid_variances = {f: v for f, v in variances.items() if v is not None}

    # If no valid variances are available, it's an unrecoverable state.
    if not valid_variances:
      raise ValueError(
          'Cannot select next stimulus: No valid variances found in hybrid '
          'audiogram. Check for repeated fit failures.')

    freq_with_max_variance = max(valid_variances, key=valid_variances.get)
    # The most informative level to test is the current threshold estimate.
    level_to_test = thresholds.get(freq_with_max_variance)

    if level_to_test is None:
      # This indicates an internal inconsistency and should be a hard error.
      raise ValueError(
          f'Cannot select next stimulus: Threshold for frequency '
          f'{freq_with_max_variance} Hz is None, despite having a valid '
          f'variance. This indicates an inconsistent audiogram structure.'
      )
    level_to_test = np.clip(level_to_test, min_level, max_level)
    if verbosity >= 2:
      print(f'Max variance found at {freq_with_max_variance:.0f} Hz. '
            f'Testing at threshold {level_to_test:.1f} dBHL.')
    return freq_with_max_variance, level_to_test


class KernelReconstructor(AudiogramReconstructor):
  """Reconstructs an audiogram using the kernel smoother."""

  def __init__(self):
    """Initializes the reconstructor, setting up grid configurations."""
    self.grid_config = adaptive_audiometry.GRID_CONFIG.copy()
    self.grid_config['min_level_dbhl'] -= 10
    self.grid_config['max_level_dbhl'] += 10
    self.grid_points_x = adaptive_audiometry.create_test_grid(
      grid_config=self.grid_config,
      standard_frequencies=STANDARD_FREQS_HZ
    )
    self.kernel_config = adaptive_audiometry.KERNEL_SMOOTHER_CONFIG.copy()
    self.final_kernel_probs = None

  def reconstruct(
      self, history: List[Tuple[float, float, bool]], verbosity: int
  ) -> Dict[str, Dict[float, Optional[float]]]:
    """Performs final analysis using the kernel smoother."""
    kernel_audiogram: Dict[float, Optional[float]] = {
        f: None for f in STANDARD_FREQS_HZ
    }
    final_kernel_probs = None
    if history:
      x_all_raw, y_all_raw = adaptive_audiometry.preprocess_data(history)
      x_all_augmented, y_all_augmented = \
          adaptive_audiometry.add_ghost_points(x_all_raw, y_all_raw,
                                               STANDARD_FREQS_HZ)

      if x_all_augmented.shape[0] > 0:
        final_kernel_probs = adaptive_audiometry.fit_kernel_smoother(
            x_all_augmented, y_all_augmented, self.grid_points_x,
            self.kernel_config, verbosity=verbosity
        )
        kernel_audiogram = adaptive_audiometry.estimate_thresholds(
            self.grid_points_x, final_kernel_probs, STANDARD_FREQS_HZ,
            verbosity=verbosity
        )
    self.final_kernel_probs = final_kernel_probs
    return {'Kernel': kernel_audiogram}


class HybridLogisticReconstructor(AudiogramReconstructor):
  """ A wrapper that reconstructs audiograms using the hybrid logistic fit."""

  def __init__(self):
    """Initializes the reconstructor."""
    pass

  def reconstruct(
      self, history: List[Tuple[float, float, bool]], verbosity: int
  ) -> Dict[str, any]:
    """
    Triggers the hybrid logistic fit and returns all resulting audiograms.

    Calls `fit_audiogram_hybrid` and packages its outputs—the final hybrid
    audiogram, the intermediate local-only and global-only fits.
    """
    (hybrid_audiogram, local_audiogram,
     global_audiogram) = fit_audiogram_hybrid(history, STANDARD_FREQS_HZ)

    return {
        'Hybrid': hybrid_audiogram,
        'Global Parametric': global_audiogram,
        'Local': local_audiogram
    }
