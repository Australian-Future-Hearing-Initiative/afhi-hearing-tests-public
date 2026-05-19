""" Module for fitting audiogram thresholds using logistic functions.

This module provides several strategies for estimating hearing thresholds from
a series of stimulus-response trials:
- **Local Fit**: `fit_audiogram_logistic` fits an independent psychometric
  curve for each frequency using a discrete grid search (no scipy).
- **Global Fit**: `fit_audiogram_logistic_parameterized` fits a
  single, smooth audiogram shape across all frequencies.
- **Hybrid Fit**: `fit_audiogram_hybrid` combines the global and local fits
  using an optimal, inverse-variance weighted average.
"""

import itertools
import numpy as np
from typing import List, Tuple, Dict, Optional
from hearing_models import hearing_level_model

# Global fit constants.
# Defines the search space for the hearing loss model coefficients.
PARAMETERIZED_FIT_BOUNDS = {
  'c1': (0, 2),
  'c2': (-1, 1),
  'c3': (-1, 1),
  'c4': (-1, 1),
}
# Defines the fixed spread of the psychometric curve for a more robust fit.
HARDCODED_SPREAD = 10.0
# Defines the variance of the global parametric model, acting as a prior.
GLOBAL_FIT_VARIANCE = 150.0

# Local fit constants.
# Defines the anchor points for the logistic fit to ensure model stability.
PRIOR_LOW_DBHL = -10.0
PRIOR_HIGH_DBHL = 90.0
# Defines the search range for the local fit relative to the anchors.
# (Matches the original curve_fit bounds: prior +/- 10dB)
GRID_MIN_DBHL = PRIOR_LOW_DBHL - 10.0
GRID_MAX_DBHL = PRIOR_HIGH_DBHL + 10.0
# Defines the search range for the spread (slope) and the number of steps.
GRID_MIN_SPREAD = 1.0  # Minimum spread value.
GRID_MAX_SPREAD = 10.0  # Maximum spread value.
GRID_NUM_SPREADS = 20  # Number of spread steps in the grid search.
# Scales the local variance estimate (Var = K * Spread^2 / N).
K_HEURISTIC = 30.0
# Defines the assumed lapse rate for human responses.
# Used to smooth labels (0 -> LAPSE_RATE, 1 -> 1-LAPSE_RATE) to prevent
# the logistic fit from collapsing to a step function (spread=0).
LAPSE_RATE = 0.02


def psychometric_logistic_curve(level_dbhl: np.ndarray,
                                threshold_dbhl: float,
                                spread: float) -> np.ndarray:
  """
  Calculates the probability of hearing using a logistic function.

  Args:
    level_dbhl: The stimulus level in dB HL.
    threshold_dbhl: The threshold (50% probability point) in dB HL.
    spread: The spread of the curve (a smaller value is steeper).

  Returns:
    The probability of hearing (0.0 to 1.0).
  """
  return 1.0 / (1.0 + np.exp((threshold_dbhl - level_dbhl) / spread))


class GlobalDictionaryFitter:
  """ A robust, solver-free fitter for the global audiogram model.
  """
  _SHARED_SHAPES: np.ndarray = np.array([])
  _CACHED_FREQS: Optional[tuple] = None

  def __init__(self, standard_freqs: list[float]):
    self.freqs = np.array(standard_freqs)
    freqs_tuple = tuple(standard_freqs)

    if GlobalDictionaryFitter._CACHED_FREQS != freqs_tuple:
      GlobalDictionaryFitter._SHARED_SHAPES = self._generate_shapes(self.freqs)
      GlobalDictionaryFitter._CACHED_FREQS = freqs_tuple

    self.shapes = GlobalDictionaryFitter._SHARED_SHAPES

  def _generate_shapes(self, freqs: np.ndarray) -> np.ndarray:
    b1, _ = hearing_level_model(freqs, np.array([1.0, 0.0, 0.0, 0.0]))
    b2, _ = hearing_level_model(freqs, np.array([0.0, 1.0, 0.0, 0.0]))
    b3, _ = hearing_level_model(freqs, np.array([0.0, 0.0, 1.0, 0.0]))
    b4, _ = hearing_level_model(freqs, np.array([0.0, 0.0, 0.0, 1.0]))
    b1, b2, b3, b4 = b1.flatten(), b2.flatten(), b3.flatten(), b4.flatten()

    c1_vals = np.linspace(0.0, 2.0, 50)
    c2_vals = np.linspace(-1.0, 1.0, 10)
    c3_vals = np.linspace(-0.5, 0.5, 5)
    c4_vals = np.linspace(-0.5, 0.5, 5)

    shapes = []
    for c1, c2, c3, c4 in itertools.product(c1_vals, c2_vals, c3_vals, c4_vals):
      curve = c1 * b1 + c2 * b2 + c3 * b3 + c4 * b4
      shapes.append(curve)
    return np.array(shapes)

  def fit(self, past_results: list[tuple[float, float, bool]]) -> dict[
    float, float]:
    if not past_results:
      raise ValueError('Cannot fit audiogram with empty results')

    trial_indices = []
    trial_levels = []
    trial_responses = []

    for freq, level, response in past_results:
      idx = np.argmin(np.abs(self.freqs - freq))
      trial_indices.append(idx)
      trial_levels.append(level)
      trial_responses.append(1.0 if response else 0.0)

    trial_indices = np.array(trial_indices)
    trial_levels = np.array(trial_levels)
    trial_responses = np.array(trial_responses)

    relevant_predictions = self.shapes[:, trial_indices]
    z = (relevant_predictions - trial_levels) / HARDCODED_SPREAD
    probs = 1.0 / (1.0 + np.exp(z))

    # Sum of Squared Errors
    total_errors = np.sum((probs - trial_responses) ** 2, axis=1)
    best_idx = np.argmin(total_errors)
    best_curve = self.shapes[best_idx]

    return dict(zip(self.freqs, best_curve))


class LocalGridFitter:
  """A solver-free fitter for the local audiogram model.

  Replaces scipy curve_fit with a discrete grid search. The search space is
  derived directly from the module constants (PRIOR_LOW_DBHL, etc).
  """
  def __init__(self):
    # 1. Define Threshold Grid based on Anchors
    # We use the same bounds as the original logic: [Low - 10, High + 10]
    # Step size of 1.0 dB is clinically sufficient.
    self.threshold_grid = np.arange(
        GRID_MIN_DBHL,
        GRID_MAX_DBHL + 1.0, # +1 for inclusive range
        1.0
    )

    # 2. Define Spread Grid
    self.spread_grid = np.linspace(GRID_MIN_SPREAD, GRID_MAX_SPREAD,
                                   GRID_NUM_SPREADS)

    # 3. Pre-calculate the cartesian product for vectorized search
    # Shapes will be (N_combinations,)
    self.t_flat = np.repeat(self.threshold_grid, len(self.spread_grid))
    self.s_flat = np.tile(self.spread_grid, len(self.threshold_grid))

  def fit(self, levels: List[float], responses: List[float]
          ) -> Tuple[float, float]:
    """ Finds the best (threshold, spread) that minimizes squared error. """
    levels_arr = np.array(levels)
    responses_arr = np.array(responses)

    # Vectorized calculation:
    # z = (Threshold - Stimulus) / Spread
    # dimensions: (N_models, N_trials)
    z = (self.t_flat[:, None] - levels_arr[None, :]) / self.s_flat[:, None]

    probs = 1.0 / (1.0 + np.exp(z))

    squared_errors = (probs - responses_arr[None, :]) ** 2
    total_errors = np.sum(squared_errors, axis=1)

    best_idx = np.argmin(total_errors)

    return self.t_flat[best_idx], self.s_flat[best_idx]


# Instantiate the fitter once to pre-compute the grid.
_LOCAL_FITTER = LocalGridFitter()


def fit_audiogram_logistic(
    past_results: List[Tuple[float, float, bool]],
    standard_freqs_hz: List[float]
) -> tuple[Dict[float, Optional[float]], Dict[float, Optional[float]],
Dict[float, Optional[float]]]:
  """Estimates audiogram thresholds using a grid search (No Scipy)."""
  if not past_results:
    raise ValueError('Cannot fit audiogram with empty results')

  estimated_audiogram = {}
  fitted_spreads = {}
  fitted_variances = {}

  for target_freq_hz in standard_freqs_hz:
    subject_levels = []
    subject_responses = []

    for freq, level, response in past_results:
      if abs(freq - target_freq_hz) < 1e-3:
        subject_levels.append(level)
        # Apply label smoothing based on lapse rate
        val = 1.0 - LAPSE_RATE if response else LAPSE_RATE
        subject_responses.append(val)

    # Add Priors (Anchors) to the data to stabilize the fit.
    # These are the same anchors used to derive the search grid boundaries.
    fit_levels = subject_levels + [PRIOR_LOW_DBHL, PRIOR_HIGH_DBHL]
    fit_responses = subject_responses + [LAPSE_RATE, 1.0 - LAPSE_RATE]

    if len(subject_levels) < 1:
      estimated_audiogram[target_freq_hz] = None
      fitted_spreads[target_freq_hz] = None
      fitted_variances[target_freq_hz] = None
      continue

    fitted_threshold, fitted_spread = _LOCAL_FITTER.fit(fit_levels,
                                                        fit_responses)

    effective_n = len(subject_levels)
    fitted_variance = K_HEURISTIC * (fitted_spread ** 2) / effective_n

    estimated_audiogram[target_freq_hz] = fitted_threshold
    fitted_spreads[target_freq_hz] = fitted_spread
    fitted_variances[target_freq_hz] = fitted_variance

  return estimated_audiogram, fitted_spreads, fitted_variances


def fit_audiogram_global(
    past_results: list[tuple[float, float, bool]],
    standard_freqs_hz: list[float]
) -> dict[float, Optional[float]]:
  """ Estimates an audiogram using the Global Dictionary Search method. """
  fitter = GlobalDictionaryFitter(standard_freqs_hz)
  return fitter.fit(past_results)


def fit_audiogram_hybrid(
    past_results: list[tuple[float, float, bool]],
    standard_freqs_hz: list[float]
) -> Tuple[
  dict[str, dict[float, Optional[float]]],
  dict[str, dict[float, Optional[float]]],
  dict[float, Optional[float]]
]:
  """ Combines local and global fits for a robust audiogram estimate."""

  local_thresholds, local_spreads, local_variances = fit_audiogram_logistic(
    past_results, standard_freqs_hz
  )
  local_audiogram = {
    'thresholds': local_thresholds,
    'spreads': local_spreads,
    'variances': local_variances
  }

  global_audiogram = fit_audiogram_global(
    past_results, standard_freqs_hz
  )

  hybrid_thresholds = {}
  hybrid_variances = {}

  for freq in standard_freqs_hz:
    t_local = local_thresholds.get(freq)
    var_local = local_variances.get(freq)
    t_global = global_audiogram.get(freq)

    if t_local is None or var_local is None:
      hybrid_thresholds[freq] = None
      hybrid_variances[freq] = None
      continue

    w_local = 1.0 / var_local
    w_global = 1.0 / GLOBAL_FIT_VARIANCE

    hybrid_t = (w_local * t_local + w_global * t_global) / (w_local + w_global)
    hybrid_var = 1.0 / (w_local + w_global)

    hybrid_thresholds[freq] = hybrid_t
    hybrid_variances[freq] = hybrid_var

  hybrid_audiogram = {
    'thresholds': hybrid_thresholds,
    'variances': hybrid_variances
  }

  return hybrid_audiogram, local_audiogram, global_audiogram
