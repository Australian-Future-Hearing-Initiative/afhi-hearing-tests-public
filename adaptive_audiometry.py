"""Adaptive functions to model audiogram and select tones.

Uses kernel smoothing to estimate the probability surface and an active
learning strategy (entropy sampling) to select the next stimulus.
"""

import numpy as np
from typing import Optional, Any

from common import PTA_MAX_LEVEL_DB_HL, PTA_MIN_LEVEL_DB_HL


# Default configuration for the stimulus grid generation.
GRID_CONFIG: dict[str, Any] = {
    'min_freq_hz': 250.0,       # Minimum frequency in Hz.
    'max_freq_hz': 8000.0,      # Maximum frequency in Hz.
    'freq_step_octaves': 0.5,   # Step size in octaves.
    'min_level_dbhl': PTA_MIN_LEVEL_DB_HL,  # Minimum level in dB HL.
    'max_level_dbhl': PTA_MAX_LEVEL_DB_HL,  # Maximum level in dB HL.
    'level_step_db': 1.0,       # Step size in dB.
}

# Default configuration for the kernel smoother. These control the smoothness
# of the probability surface in each dimension.
KERNEL_SMOOTHER_CONFIG: dict[str, Any] = {
    'log_freq_lengthscale': 0.5,  # Lengthscale for log2(freq) dimension.
    'level_lengthscale': 10.0,  # Lengthscale for level (dBHL) dimension.
    'epsilon': 1e-9  # Small value to prevent division by zero.
}

# Default configuration for the initial structured testing phase.
INITIAL_PHASE_CONFIG: dict[str, Any] = {
    'target_freq_hz': 1000.0,     # Frequency for initial descent.
    'start_level_dbhl': 60.0,     # Starting level for 1 kHz descent.
    'descent_step_db': 10.0,      # Step size for decreasing level at 1 kHz.
    'sweep_level_step_db': 15.0,  # Step size for changing level during sweep.
    'min_level_dbhl': PTA_MIN_LEVEL_DB_HL,  # Minimum level for descent.
    'max_level_dbhl': PTA_MAX_LEVEL_DB_HL,  # Maximum level for sweep.
}

# Define ghost point constants. These provide prior knowledge about levels that
# are known to be "heard" or "not heard" at the extremes of the level range.
GHOST_POINT_MIN_LEVEL = -15.0  # Assume no one can hear this.
GHOST_POINT_MAX_LEVEL = 85.0   # Assume everyone can hear this.

def preprocess_data(
    past_results: list[tuple[float, float, bool]]
) -> tuple[np.ndarray, np.ndarray]:
  """Convert trial results list to numerical format (X, y 0/1).

  Handles transformations like log-frequency scaling. Maps boolean
  responses (True=heard, False=not heard) to 1.0 / 0.0.

  Args:
    past_results: List of (frequency_hz, level_dbhl, response) tuples.

  Returns:
    Tuple containing:
      - X: NumPy array (n_trials, 2) with features [log2(freq), level].
      - y: NumPy array (n_trials,) with numerical responses (1.0 / 0.0).
  """
  if not past_results:
    return np.empty((0, 2)), np.empty((0,))

  features_list = []
  responses_list = []
  for freq_hz, level_dbhl, response in past_results:
    log2_freq = np.log2(freq_hz)
    features_list.append([log2_freq, level_dbhl])
    responses_list.append(1.0 if response else 0.0)

  X = np.array(features_list)  # pylint: disable=invalid-name
  y = np.array(responses_list)
  return X, y

def add_ghost_points(x_data: np.ndarray, y_data: np.ndarray,
                     standard_freqs: list[float]
                     ) -> tuple[np.ndarray, np.ndarray]:
  """
  Augments training data with 'ghost' points to enforce prior knowledge.

  Adds points representing strong beliefs about hearing at very low and very
  high dBHL levels across all standard frequencies. This helps to regularize
  the model, especially when data is sparse.

  Args:
    x_data: The original frequency (in log2(Hz)) and level data.
    y_data: The original response data (heard/not heard).
    standard_freqs: The list of standard audiometric frequencies in Hz.

  Returns:
    A tuple of (x_augmented, y_augmented) with the new points added.
  """
  ghost_points_x = []
  ghost_points_y = []
  log2_standard_freqs = np.log2(standard_freqs)

  for freq_log2 in log2_standard_freqs:
    # Assume tone is always heard at a very high level (e.g., 120 dBHL).
    ghost_points_x.append([freq_log2, GHOST_POINT_MAX_LEVEL])
    ghost_points_y.append(1.0) # 1.0 represents 'heard'.

    # Assume tone is never heard at a very low level (e.g., -20 dBHL).
    ghost_points_x.append([freq_log2, GHOST_POINT_MIN_LEVEL])
    ghost_points_y.append(0.0) # 0.0 represents 'not heard'.

  x_augmented = (np.vstack([x_data, np.array(ghost_points_x)]) if
                 x_data.size else np.array(ghost_points_x))
  y_augmented = (np.concatenate([y_data, np.array(ghost_points_y)]) if
                 y_data.size else np.array(ghost_points_y))

  return x_augmented, y_augmented


def fit_kernel_smoother(
    X_train: np.ndarray,  # pylint: disable=invalid-name
    y_train: np.ndarray, # Expects 0/1 input
    grid_points_x: np.ndarray,
    kernel_config: dict[str, Any],
    verbosity: int
) -> np.ndarray:
  """Predict probabilities on a grid using kernel smoothing (Nadaraya-Watson).

  Uses an anisotropic RBF kernel.

  Args:
      X_train: Training input data (n_train, 2) [log2(freq), level].
      y_train: Training responses (n_train,) [0.0 / 1.0].
      grid_points_x: Points to predict probabilities at (n_grid, 2).
      kernel_config: Dictionary with kernel parameters, expecting:
                     'log_freq_lengthscale', 'level_lengthscale', 'epsilon'.
      verbosity: The verbosity level for console output.

  Returns:
      NumPy array (n_grid,) of predicted probabilities.
  """
  n_grid = grid_points_x.shape[0]
  n_train = X_train.shape[0]
  predicted_probs = np.zeros(n_grid)
  # Get kernel parameters.
  ls_freq = kernel_config.get('log_freq_lengthscale', 0.5)
  ls_level = kernel_config.get('level_lengthscale', 5.0)
  epsilon = kernel_config.get('epsilon', 1e-9)

  if n_train == 0:
    if verbosity >= 0:
      print('Warning: No data provided for kernel smoothing. Returning 0.5.')
    return np.full(n_grid, 0.5)

  # Pre-calculate squared lengthscales for efficiency.
  ls_freq_sq = ls_freq**2
  ls_level_sq = ls_level**2

  # Iterate through each grid point to predict.
  for i in range(n_grid):
    grid_point = grid_points_x[i, :] # Current grid point (1, 2)

    # Calculate squared distances (anisotropic).
    diff = grid_point - X_train
    sq_dist_freq = (diff[:, 0]**2) / ls_freq_sq
    sq_dist_level = (diff[:, 1]**2) / ls_level_sq
    sq_distances = sq_dist_freq + sq_dist_level # (n_train,)

    # Calculate RBF kernel weights.
    weights = np.exp(-0.5 * sq_distances) # (n_train,)

    # Calculate weighted average (Nadaraya-Watson estimate).
    sum_weights = np.sum(weights)

    if sum_weights < epsilon:
      # If grid point is too far from all training points, predict 0.5.
      predicted_probs[i] = 0.5
    else:
      weighted_sum_y = np.sum(weights * y_train)
      predicted_probs[i] = weighted_sum_y / sum_weights

  # Clip probabilities just in case of numerical issues.
  predicted_probs = np.clip(predicted_probs, epsilon, 1.0 - epsilon)
  return predicted_probs

def create_test_grid(grid_config: dict[str, Any],
                     standard_frequencies: Optional[list[float]] = None
                     ) -> np.ndarray:
  """Generate the grid of candidate stimulus points in the feature space.

  Creates a 2D grid (log2(frequency), level_dbhl) based on the
  ranges and steps defined in grid_config. The frequency dimension is handled
  in one of two ways:
  1. If `standard_frequencies` is provided and not empty, only these frequencies
     are used for the grid.
  2. If `standard_frequencies` is None or empty, frequencies are generated
     based on the 'min_freq_hz', 'max_freq_hz', and 'freq_step_octaves'
     in the `grid_config`.

  Args:
    grid_config: Dictionary with grid parameters.
                 See DEFAULT_GRID_CONFIG for expected keys.
    standard_frequencies: Optional list of standard frequencies (Hz) to use
                          exclusively for the frequency dimension.

  Returns:
    NumPy array (n_grid_points, 2) representing candidate points in the
    feature space [log2(freq), level_dbhl].
  """
  # Determine the frequency points (in log2 space).
  if standard_frequencies and len(standard_frequencies) > 0:
    # Use only the provided standard frequencies.
    freq_points_log2 = np.unique(np.log2(standard_frequencies))
  else:
    # Generate frequencies based on grid_config range and step.
    min_freq_log2 = np.log2(grid_config['min_freq_hz'])
    max_freq_log2 = np.log2(grid_config['max_freq_hz'])
    # Ensure at least one step if min and max are the same.
    if max_freq_log2 == min_freq_log2:
      num_steps = 1
    else:
      num_steps = int(np.round(
          (max_freq_log2 - min_freq_log2) / grid_config['freq_step_octaves']
          )) + 1
    freq_points_log2 = np.linspace(min_freq_log2, max_freq_log2, num_steps)

  # Generate level points based on grid_config.
  min_level = grid_config['min_level_dbhl']
  max_level = grid_config['max_level_dbhl']
  level_step_db = grid_config['level_step_db']
  # Use arange for exact steps. Add epsilon to include max_level if it's a step.
  level_points = np.arange(min_level, max_level + 1e-9, level_step_db)

  log_freq_grid, level_grid = np.meshgrid(freq_points_log2, level_points)
  grid_points = np.vstack([log_freq_grid.ravel(), level_grid.ravel()]).T
  return grid_points

def _find_next_stimulus(
    grid_points_x: np.ndarray,
    acquisition_values: np.ndarray
) -> np.ndarray:
  """Select the grid point that maximizes the acquisition function.

  Assumes non-empty and compatible inputs.

  Args:
    grid_points_x: Candidate points in the feature space (n_points, 2).
    acquisition_values: Corresponding acquisition function values (n_points,).

  Returns:
    The single grid point (1D NumPy array, shape (2,)) maximizing
    acquisition value.
  """
  best_index = np.argmax(acquisition_values)
  next_stimulus_x = grid_points_x[best_index, :]
  return next_stimulus_x

def _postprocess_stimulus(stimulus_x: np.ndarray) -> tuple[float, float]:
  """Convert the selected stimulus from feature space to physical units.

  Handles inverse transformations (e.g., log-frequency to Hz).

  Args:
    stimulus_x: Selected stimulus in feature space (e.g., [log2(freq), level]).
                Expected to be a 1D NumPy array of shape (2,).

  Returns:
    Tuple (frequency_hz, level_dbhl).

  Raises:
      ValueError: If stimulus_x does not have the expected shape.
  """
  if stimulus_x.shape != (2,):
    raise ValueError(
        f"Expected stimulus_x to have shape (2,), got {stimulus_x.shape}"
    )
  log2_freq = stimulus_x[0]
  level_dbhl = stimulus_x[1]
  frequency_hz = np.power(2.0, log2_freq)
  # Round to nearest integer to avoid floating point issues with keys.
  rounded_frequency_hz = round(frequency_hz)
  return float(rounded_frequency_hz), float(level_dbhl)

def _binary_entropy(p: np.ndarray) -> np.ndarray:
  """Calculate binary entropy H(p) = -p*log2(p) - (1-p)*log2(1-p).

  Args:
      p: NumPy array of probabilities (between 0 and 1).

  Returns:
      NumPy array of entropy values.
  """
  epsilon = 1e-9 # Small constant.
  p = np.clip(p, epsilon, 1.0 - epsilon)
  return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)

def run_adaptive_test_step(
    past_results: list[tuple[float, float, bool]],
    standard_freqs: list[float],
    initial_phase_config: Optional[dict[str, Any]] = None,
    kernel_config: Optional[dict[str, Any]] = None,
    verbosity: int = 1
) -> tuple[tuple[float, float], dict[float, Optional[float]]]:
  """Determine the next stimulus, incorporating an initial structured phase.

  Manages the test state:
    1.  Starts with a 1 kHz descent to find the first threshold.
    2.  Sweeps across other frequencies to get initial estimates.
    3.  Switches to a fully adaptive mode using a kernel smoother and entropy-
        based sampling to select the most informative next stimulus.

  Args:
    past_results: List of previous trials [(freq, level, response), ...].
    standard_freqs: List of standard frequencies to test (Hz).
    initial_phase_config: Configuration for the initial phase. Uses defaults.
    kernel_config: Configuration for the kernel smoother. Uses defaults.
    verbosity: The verbosity level for console output.

  Returns:
    A tuple containing:
      - (next_frequency_hz, next_level_dbhl): Stimulus for the next trial.
      - dict[float, Optional[float]]: Current estimated audiogram thresholds.

  Raises:
      ValueError: If standard_freqs is empty.
  """
  if not standard_freqs:
    raise ValueError('standard_freqs cannot be empty.')

  # Handle default configurations.
  final_initial_config = INITIAL_PHASE_CONFIG.copy()
  if initial_phase_config:
    final_initial_config.update(initial_phase_config)

  final_kernel_config = KERNEL_SMOOTHER_CONFIG.copy()
  if kernel_config:
    final_kernel_config.update(kernel_config)

  # --- Determine Current Test Phase ---
  target_freq = final_initial_config['target_freq_hz']
  min_level = final_initial_config['min_level_dbhl']
  max_level = final_initial_config['max_level_dbhl']
  sweep_step_db = final_initial_config['sweep_level_step_db']
  descent_step = final_initial_config['descent_step_db']
  start_level = final_initial_config['start_level_dbhl']

  # Get the results just for the target frequency.
  target_freq_results = [r for r in past_results if r[0] == target_freq]

  # Find results where the tone was heard vs. not heard.
  target_freq_heard = [r for r in target_freq_results if r[2]]
  target_freq_not_heard = [r for r in target_freq_results if not r[2]]

  estimated_audiogram: dict[float, Optional[float]] = {
      f: None for f in standard_freqs
  }

  # --- State Machine for the test ---
  # Phase 1: Initial 1 kHz Descent.
  if not target_freq_not_heard:
    if verbosity >= 3:
      print('Phase 1: Initial Descent')
    last_heard_level = -np.inf # Level below minimum.
    if target_freq_heard:
      # Find the minimum level where the tone was heard.
      last_heard_level = min(r[1] for r in target_freq_heard)

    # Determine next level: Go down from last heard level or start level.
    start_or_last_heard = (start_level if last_heard_level == -np.inf else
                           last_heard_level)
    next_level = start_or_last_heard - descent_step
    next_level = np.clip(next_level, min_level, max_level)

    # Check if we are "stuck" at the minimum level.
    already_heard_min_level = any(
        r[1] == min_level for r in target_freq_heard)
    if next_level == min_level and already_heard_min_level:
      # Stuck condition: Heard min level, proposing min level again.
      # Do nothing here; let execution fall through to Phase 2 check below.
      if verbosity >= 3:
        print(f'Minimum level {min_level} dBHL heard.')
      pass
    else:
      # Normal Phase 1 operation or proposing min_level for the first time.
      if verbosity >= 3:
        print(f"Proposing {target_freq} Hz at {next_level} dBHL (Descent)")
      # Return stimulus and empty audiogram during initial descent.
      return (target_freq, next_level), estimated_audiogram

  # Phase 2: Initial Broad Sweep.
  # Get frequencies that have not been tested at all in the sweep.
  tested_freqs = {res[0] for res in past_results if res[0] != target_freq}
  remaining_sweep_freqs = [
    f for f in standard_freqs if f != target_freq and f not in tested_freqs]

  if remaining_sweep_freqs:
    if verbosity >= 3:
      print('Phase 2: Broad Sweep')
    next_sweep_freq = remaining_sweep_freqs[0] # Take the first untested.

    # Determine the level adaptively.
    # Start from the threshold of the most recently tested sweep frequency.
    tested_sweep_freqs = sorted(list(tested_freqs))
    if not tested_sweep_freqs:
      # If no other freq has been tested, start from the 1kHz threshold.
      if not target_freq_heard:
        # Should not happen if target_freq_not_heard is True, but as a fallback.
        next_level = start_level
      else:
        next_level = min(r[1] for r in target_freq_heard)
    else:
      prev_sweep_freq = tested_sweep_freqs[-1]
      prev_results = [r for r in past_results if r[0] == prev_sweep_freq]
      if not prev_results:
        # Should not happen if tested_sweep_freqs is not empty, but fallback.
        if verbosity >= 0:
          print('Warning: No results found for previous sweep '
                f'freq {prev_sweep_freq}. Using start_level.')
        next_level = start_level
      else:
        prev_level = prev_results[-1][1]
        prev_response_heard = prev_results[-1][2]
        # Simple heuristic: if heard, go down; if not, go up.
        level_step = sweep_step_db if prev_response_heard else -sweep_step_db
        next_level = prev_level - level_step

    # Apply bounds to the calculated level.
    next_level = np.clip(next_level, min_level, max_level)

    if verbosity >= 3:
      print(f'Proposing {next_sweep_freq} Hz at {next_level:.1f} dBHL (Sweep)')
    # Return stimulus and empty audiogram during sweep.
    return (next_sweep_freq, next_level), estimated_audiogram

  # Phase 3: Adaptive Phase (Kernel Smoothing).
  if verbosity >= 3:
    print('Phase 3: Adaptive Kernel Smoothing')

  # --- Run kernel smoothing and estimate audiogram ---
  # Preprocess data
  x_train, y_train = preprocess_data(past_results)
  x_train, y_train = add_ghost_points(x_train, y_train, standard_freqs)

  # Check for sufficient data before smoothing.
  if x_train.shape[0] < 1:
    if verbosity >= 0:
      print('Warning: Insufficient data for adaptive phase. Returning default.')
    default_level = start_level / 2.0 # Arbitrary mid-level guess.
    next_stimulus = (float(target_freq), float(default_level))
    return next_stimulus, estimated_audiogram # Return default None audiogram

  # Define candidate grid.
  grid_points_x = create_test_grid(
      grid_config=GRID_CONFIG,
      standard_frequencies=standard_freqs)
  # Fit smoother and get probabilities.
  predicted_probs = fit_kernel_smoother(x_train, y_train, grid_points_x,
                                        final_kernel_config,
                                        verbosity=verbosity)
  # Estimate audiogram from probabilities.
  estimated_audiogram = estimate_thresholds(
      grid_points_x, predicted_probs, standard_freqs, verbosity=verbosity
  )
  # Calculate acquisition function (entropy sampling).
  acquisition_values = _binary_entropy(predicted_probs)
  # Select best next stimulus based on acquisition.
  next_stimulus_x = _find_next_stimulus(grid_points_x, acquisition_values)
  # Postprocess stimulus.
  next_freq_hz, next_level_dbhl = _postprocess_stimulus(next_stimulus_x)
  next_stimulus = (next_freq_hz, next_level_dbhl)
  # Return next stimulus and the estimated audiogram.
  return next_stimulus, estimated_audiogram

def estimate_thresholds(
    grid_points_x: np.ndarray,
    predicted_probs: np.ndarray,
    standard_freqs: list[float],
    verbosity: int
) -> dict[float, Optional[float]]:
  """Estimates thresholds at standard frequencies from predicted probabilities.

  Args:
    grid_points_x: The grid of (log2_freq, level) points.
    predicted_probs: Predicted probabilities at each grid point.
    standard_freqs: List of standard freqs (Hz) to estimate thresholds for.
    verbosity: The verbosity level for console output.
 
  Returns:
    Dictionary mapping standard frequencies to estimated thresholds (dB HL)
    or None if estimation fails (e.g., probability doesn't cross 0.5).
  """
  estimated_audiogram: dict[float, Optional[float]] = {}
  log2_standard_freqs = np.log2(standard_freqs)
  unique_log2_freqs = np.unique(grid_points_x[:, 0])
  tol = 1e-6

  for std_freq, log2_std_freq in zip(standard_freqs, log2_standard_freqs):
    closest_grid_log2_freq_idx = np.argmin(
        np.abs(unique_log2_freqs - log2_std_freq))
    closest_grid_log2_freq = unique_log2_freqs[closest_grid_log2_freq_idx]

    if abs(closest_grid_log2_freq - log2_std_freq) > tol:
      if verbosity >= 0:
        print(
            f'Warning: No grid point close enough to {std_freq} Hz. '
            f'Skipping threshold estimation.'
        )
      estimated_audiogram[std_freq] = None
      continue

    freq_indices = np.where(
        np.abs(grid_points_x[:, 0] - closest_grid_log2_freq) < tol)[0]
    if len(freq_indices) == 0:
      estimated_audiogram[std_freq] = None
      continue

    levels_at_freq = grid_points_x[freq_indices, 1]
    probs_at_freq = predicted_probs[freq_indices]
    sort_indices = np.argsort(levels_at_freq)
    levels_sorted = levels_at_freq[sort_indices]
    probs_sorted = probs_at_freq[sort_indices]

    if len(probs_sorted) == 0:
      estimated_audiogram[std_freq] = None
      continue

    abs_diff_from_half = np.abs(probs_sorted - 0.5)
    closest_idx = np.argmin(abs_diff_from_half)
    estimated_threshold = levels_sorted[closest_idx]
    estimated_audiogram[std_freq] = float(estimated_threshold)

  return estimated_audiogram
