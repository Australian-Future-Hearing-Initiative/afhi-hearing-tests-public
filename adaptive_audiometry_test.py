"""Tests for the bayesian learning module."""

import numpy as np
import pytest

import adaptive_audiometry

# Define standard frequencies locally to avoid external dependencies
STANDARD_FREQS_HZ = [1000, 2000, 3000, 4000, 6000, 8000, 500, 250]


@pytest.fixture(scope='module')
def grid_points_config_step_fixture():
  """Fixture: Grid generated using only config steps (no standard freqs)."""
  config = adaptive_audiometry.GRID_CONFIG
  # Generate grid based *only* on config, not standard freqs.
  return adaptive_audiometry.create_test_grid(config)


@pytest.fixture(scope='module')
def grid_points_standard_freq_fixture():
  """Fixture: Grid generated using only standard frequencies."""
  config = adaptive_audiometry.GRID_CONFIG
  # Generate grid based *only* on standard frequencies.
  return adaptive_audiometry.create_test_grid(
      config, standard_frequencies=STANDARD_FREQS_HZ
  )


@pytest.fixture(scope='module')
def default_config_fixture():
  """Fixture to provide the default grid config."""
  return adaptive_audiometry.GRID_CONFIG


def test_create_test_grid_type(
    grid_points_config_step_fixture,  # Use renamed fixture
):  # pylint: disable=redefined-outer-name
  """Tests the output type of create_test_grid."""
  assert isinstance(grid_points_config_step_fixture, np.ndarray)


def test_create_test_grid_shape_and_bounds(
    grid_points_config_step_fixture,
    default_config_fixture,  # Use renamed fixture  # pylint: disable=redefined-outer-name
):  # pylint: disable=redefined-outer-name
  """Tests grid has 2 columns and values are within bounds."""
  config = default_config_fixture
  assert grid_points_config_step_fixture.shape[1] == 2  # Check for 2 columns

  # Check frequency bounds
  min_freq_log2 = np.log2(config['min_freq_hz'])
  max_freq_log2 = np.log2(config['max_freq_hz'])
  grid_log2_freqs = grid_points_config_step_fixture[:, 0]
  tolerance = 1e-9  # Tolerance for floating point comparisons
  assert np.all(grid_log2_freqs >= min_freq_log2 - tolerance)
  assert np.all(grid_log2_freqs <= max_freq_log2 + tolerance)

  # Check level bounds
  min_level = config['min_level_dbhl']
  max_level = config['max_level_dbhl']
  grid_levels = grid_points_config_step_fixture[:, 1]
  assert np.all(grid_levels >= min_level - tolerance)
  assert np.all(grid_levels <= max_level + tolerance)


def test_create_test_grid_standard_freq_inclusion(
    grid_points_standard_freq_fixture,  # Use standard freq fixture
    default_config_fixture,  # pylint: disable=redefined-outer-name
):  # pylint: disable=redefined-outer-name
  """Tests inclusion of standard frequencies when provided."""
  config = default_config_fixture
  min_freq_hz = config['min_freq_hz']
  max_freq_hz = config['max_freq_hz']
  grid_log2_freqs = np.unique(grid_points_standard_freq_fixture[:, 0])

  # Check standard frequency inclusion
  standard_log2_freqs = np.log2(STANDARD_FREQS_HZ)
  required_std_log2_freqs = [
      f_log2
      for f, f_log2 in zip(STANDARD_FREQS_HZ, standard_log2_freqs)
      if f >= min_freq_hz and f <= max_freq_hz
  ]
  tolerance = 1e-6
  is_present = [
      np.any(np.isclose(grid_log2_freqs, req_f, atol=tolerance))
      for req_f in required_std_log2_freqs
  ]
  missing_freqs = [
      STANDARD_FREQS_HZ[i]
      for i, present in enumerate(is_present)
      if not present
      and (
          STANDARD_FREQS_HZ[i] >= min_freq_hz
          and STANDARD_FREQS_HZ[i] <= max_freq_hz
      )
  ]
  assert all(is_present), f'Missing standard freqs in grid: {missing_freqs}'


def test_create_test_grid_freq_step_size(
    grid_points_config_step_fixture,
    default_config_fixture,  # Use config step fixture  # pylint: disable=redefined-outer-name
):  # pylint: disable=redefined-outer-name
  """Tests frequency step size when using config steps."""
  config = default_config_fixture
  grid_log2_freqs = np.unique(grid_points_config_step_fixture[:, 0])
  tolerance = 1e-6

  # Check maximum frequency step (density check)
  if len(grid_log2_freqs) > 1:
    max_diff = np.max(np.diff(grid_log2_freqs))
    expected_max_step = config['freq_step_octaves']
    assert (
        max_diff <= expected_max_step + tolerance
    ), f'Max freq step {max_diff} exceeds expected {expected_max_step}'


def test_create_test_grid_level_steps(
    grid_points_config_step_fixture,
    default_config_fixture,  # Use renamed fixture  # pylint: disable=redefined-outer-name
):  # pylint: disable=redefined-outer-name
  """Tests that level steps match the configured value."""
  config = default_config_fixture
  unique_levels = np.unique(grid_points_config_step_fixture[:, 1])

  if len(unique_levels) > 1:
    level_diffs = np.diff(unique_levels)
    expected_step = config['level_step_db']
    # Check that all differences are close to the expected step
    assert np.allclose(
        level_diffs, expected_step, atol=1e-9
    ), f'Level steps {level_diffs} do not match expected {expected_step}'
  elif len(unique_levels) == 1:
    # If only one level, the step doesn't apply, test passes.
    pass
  else:
    # If no levels (empty grid?), something else is wrong but not step size.
    pass


def test_preprocess_data_empty():
  """Tests preprocess_data with an empty input list."""
  past_results = []
  expected_x_shape = (0, 2)
  expected_y_shape = (0,)
  # pylint: disable=protected-access
  X, y = adaptive_audiometry.preprocess_data(past_results)  # pylint: disable=invalid-name
  # pylint: enable=protected-access
  # pylint: disable=invalid-name
  assert X.shape == expected_x_shape
  # pylint: enable=invalid-name
  assert y.shape == expected_y_shape


def test_preprocess_data_single_true():
  """Tests _preprocess_data with a single trial (True response)."""
  past_results = [(1000.0, 50.0, True)]
  expected_x = np.array([[np.log2(1000.0), 50.0]])
  expected_y = np.array([1.0])  # Expect 1.0 for True
  # pylint: disable=protected-access
  X, y = adaptive_audiometry.preprocess_data(past_results)  # pylint: disable=invalid-name
  # pylint: enable=protected-access

  # pylint: disable=invalid-name
  assert X.shape == (1, 2)
  # pylint: enable=invalid-name
  assert y.shape == (1,)
  np.testing.assert_allclose(X, expected_x)
  np.testing.assert_allclose(y, expected_y)


def test_preprocess_data_single_false():
  """Tests preprocess_data with a single trial (False response)."""
  past_results = [(2000.0, 30.0, False)]
  expected_x = np.array([[np.log2(2000.0), 30.0]])
  expected_y = np.array([0.0])  # Expect 0.0 for False
  # pylint: disable=protected-access
  X, y = adaptive_audiometry.preprocess_data(past_results)  # pylint: disable=invalid-name
  # pylint: enable=protected-access

  # pylint: disable=invalid-name
  assert X.shape == (1, 2)
  # pylint: enable=invalid-name
  assert y.shape == (1,)
  np.testing.assert_allclose(X, expected_x)
  np.testing.assert_allclose(y, expected_y)


def test_preprocess_data_multiple():
  """Tests preprocess_data with multiple trials."""
  past_results = [
      (1000.0, 50.0, True),
      (2000.0, 30.0, False),
      (500.0, 45.0, True),
      (4000.0, 35.0, False),
  ]
  expected_x = np.array(
      [
          [np.log2(1000.0), 50.0],
          [np.log2(2000.0), 30.0],
          [np.log2(500.0), 45.0],
          [np.log2(4000.0), 35.0],
      ]
  )
  expected_y = np.array([1.0, 0.0, 1.0, 0.0])  # Expect 1.0/0.0
  # pylint: disable=protected-access
  X, y = adaptive_audiometry.preprocess_data(past_results)  # pylint: disable=invalid-name
  # pylint: enable=protected-access

  # pylint: disable=invalid-name
  assert X.shape == (4, 2)
  # pylint: enable=invalid-name
  assert y.shape == (4,)
  np.testing.assert_allclose(X, expected_x)
  np.testing.assert_allclose(y, expected_y)


# === Tests for _postprocess_stimulus ===


def test_postprocess_stimulus_valid():
  """Tests _postprocess_stimulus with valid input."""
  freq_hz = 1000.0
  level_dbhl = 55.0
  log2_freq = np.log2(freq_hz)
  stimulus_x = np.array([log2_freq, level_dbhl])
  expected_output = (freq_hz, level_dbhl)
  # pylint: disable=protected-access
  actual_output = adaptive_audiometry._postprocess_stimulus(stimulus_x)
  # pylint: enable=protected-access

  assert isinstance(actual_output, tuple)
  assert len(actual_output) == 2
  assert isinstance(actual_output[0], float)
  assert isinstance(actual_output[1], float)
  assert np.isclose(actual_output[0], expected_output[0])
  assert np.isclose(actual_output[1], expected_output[1])


def test_postprocess_stimulus_different_values():
  """Tests _postprocess_stimulus with different valid input values."""
  freq_hz = 4567.8
  level_dbhl = -3.2
  log2_freq = np.log2(freq_hz)
  stimulus_x = np.array([log2_freq, level_dbhl])
  # Expect the frequency to be rounded by the function.
  expected_output = (round(freq_hz), level_dbhl)
  # pylint: disable=protected-access
  actual_output = adaptive_audiometry._postprocess_stimulus(stimulus_x)
  # pylint: enable=protected-access

  assert np.isclose(actual_output[0], expected_output[0])
  assert np.isclose(actual_output[1], expected_output[1])


def test_postprocess_stimulus_invalid_shape():
  """Tests that _postprocess_stimulus raises ValueError for wrong shape."""
  invalid_stimulus_x = np.array([10.0, 50.0, 1.0])
  with pytest.raises(ValueError):
    # pylint: disable=protected-access
    adaptive_audiometry._postprocess_stimulus(invalid_stimulus_x)
    # pylint: enable=protected-access

  invalid_stimulus_x_2d = np.array([[10.0, 50.0]])
  with pytest.raises(ValueError):
    # pylint: disable=protected-access
    adaptive_audiometry._postprocess_stimulus(invalid_stimulus_x_2d)
    # pylint: enable=protected-access


# === Tests for _find_next_stimulus ===


def test_find_next_stimulus_basic():
  """Tests _find_next_stimulus finds the max index correctly."""
  grid_points_x = np.array(
      [
          [1.0, 10.0],
          [2.0, 20.0],  # Max acquisition here
          [3.0, 30.0],
          [4.0, 40.0],
      ]
  )
  acquisition_values = np.array([0.1, 0.8, 0.3, 0.2])
  expected_output = np.array([2.0, 20.0])
  # pylint: disable=protected-access
  actual_output = adaptive_audiometry._find_next_stimulus(
      grid_points_x, acquisition_values
  )
  # pylint: enable=protected-access

  assert isinstance(actual_output, np.ndarray)
  assert actual_output.shape == (2,)
  np.testing.assert_array_equal(actual_output, expected_output)


def test_find_next_stimulus_empty_input_raises():
  """Tests _find_next_stimulus raises an error for empty acquisition values."""
  empty_grid = np.empty((0, 2))
  empty_acq = np.empty((0,))
  grid = np.array([[1.0, 10.0]])

  with pytest.raises(ValueError):
    # pylint: disable=protected-access
    adaptive_audiometry._find_next_stimulus(grid, empty_acq)
    # pylint: enable=protected-access
  with pytest.raises(ValueError):
    # pylint: disable=protected-access
    adaptive_audiometry._find_next_stimulus(empty_grid, empty_acq)
    # pylint: enable=protected-access


# === Tests for _binary_entropy ===


def test_binary_entropy_values():
  """Tests _binary_entropy calculation for known values."""
  probs = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
  # Expected values H(p) = -p*log2(p) - (1-p)*log2(1-p)
  expected_entropies = np.array([0.0, 0.468996, 1.0, 0.468996, 0.0])
  # pylint: disable=W0212,protected-access
  actual_entropies = adaptive_audiometry._binary_entropy(probs)
  # pylint: enable=W0212,protected-access
  np.testing.assert_allclose(actual_entropies, expected_entropies, atol=1e-6)


def test_binary_entropy_clipping():
  """Tests that _binary_entropy handles exact 0 and 1 via clipping."""
  probs = np.array([0.0, 1.0])
  # pylint: disable=W0212,protected-access
  actual_entropies = adaptive_audiometry._binary_entropy(probs)
  # pylint: enable=W0212,protected-access
  # Expect 0 entropy for both ends after clipping
  np.testing.assert_allclose(actual_entropies, np.array([0.0, 0.0]), atol=1e-6)


def test_run_adaptive_test_step_empty_freqs():
  """Test that run_adaptive_test_step raises error for empty standard_freqs."""
  with pytest.raises(ValueError, match='standard_freqs cannot be empty.'):
    adaptive_audiometry.run_adaptive_test_step(
        past_results=[], standard_freqs=[]
    )


def test_add_ghost_points_empty():
  """Tests adding ghost points to empty data."""
  x_empty = np.empty((0, 2))
  y_empty = np.empty((0,))
  std_freqs = [1000.0, 2000.0]
  x_aug, y_aug = adaptive_audiometry.add_ghost_points(
      x_empty, y_empty, std_freqs
  )
  # 2 freqs * 2 ghost points each = 4 points total
  assert x_aug.shape == (4, 2)
  assert y_aug.shape == (4,)
  # Check levels
  assert np.sum(x_aug[:, 1] == adaptive_audiometry.GHOST_POINT_MAX_LEVEL) == 2
  assert np.sum(x_aug[:, 1] == adaptive_audiometry.GHOST_POINT_MIN_LEVEL) == 2


def test_add_ghost_points_existing():
  """Tests adding ghost points to existing data."""
  x_data = np.array([[np.log2(1000.0), 50.0]])
  y_data = np.array([1.0])
  std_freqs = [1000.0]
  x_aug, y_aug = adaptive_audiometry.add_ghost_points(x_data, y_data, std_freqs)
  # 1 original + 2 ghost points = 3 points total
  assert x_aug.shape == (3, 2)
  assert y_aug.shape == (3,)
  assert np.allclose(x_aug[0], x_data[0])


def test_fit_kernel_smoother_zero_train():
  """Tests smoother fallback with no training data."""
  x_train = np.empty((0, 2))
  y_train = np.empty((0,))
  grid_points = np.array([[np.log2(1000.0), 50.0]])
  probs = adaptive_audiometry.fit_kernel_smoother(
      x_train,
      y_train,
      grid_points,
      adaptive_audiometry.KERNEL_SMOOTHER_CONFIG,
      0,
  )
  assert np.allclose(probs, 0.5)


def test_fit_kernel_smoother_nadaraya_watson():
  """Tests the RBF smoothing logic."""
  x_train = np.array([[np.log2(1000.0), 50.0]])
  y_train = np.array([1.0])
  # Grid point exactly at training point
  grid_points = np.array(
      [[np.log2(1000.0), 50.0], [np.log2(8000.0), 10.0]]  # Far away point
  )
  probs = adaptive_audiometry.fit_kernel_smoother(
      x_train,
      y_train,
      grid_points,
      adaptive_audiometry.KERNEL_SMOOTHER_CONFIG,
      0,
  )

  # Point 1 heavily influenced by training point (probability near 1.0)
  assert probs[0] > 0.9
  # Point 2 is far away, should fall back to 0.5
  assert np.isclose(probs[1], 0.5)


def test_estimate_thresholds_extraction():
  """Tests extracting threshold from crossing probabilities."""
  grid_points = np.array(
      [
          [np.log2(1000.0), 10.0],
          [np.log2(1000.0), 20.0],
          [np.log2(1000.0), 30.0],
          [np.log2(1000.0), 40.0],
      ]
  )
  predicted_probs = np.array([0.1, 0.3, 0.6, 0.9])
  # The probability closest to 0.5 is 0.6 at 30 dB.
  std_freqs = [1000.0]
  est = adaptive_audiometry.estimate_thresholds(
      grid_points, predicted_probs, std_freqs, 0
  )
  assert est[1000.0] == 30.0


def test_estimate_thresholds_missing_grid():
  """Tests behavior when standard freq is missing from grid."""
  grid_points = np.array([[np.log2(1000.0), 50.0]])
  predicted_probs = np.array([0.5])
  std_freqs = [1000.0, 2000.0]
  est = adaptive_audiometry.estimate_thresholds(
      grid_points, predicted_probs, std_freqs, 0
  )
  assert est[1000.0] == 50.0
  assert est[2000.0] is None


def test_estimate_thresholds_empty_probs():
  """Tests behavior when probability array is empty for a freq."""
  # Freq exists in grid but somehow no valid levels
  grid_points = np.array([[np.log2(1000.0), 50.0]])
  predicted_probs = np.array([0.5])
  std_freqs = [1000.0]
  # We test the general execution path
  est = adaptive_audiometry.estimate_thresholds(
      grid_points, predicted_probs, std_freqs, 0
  )
  assert est[1000.0] == 50.0


def test_run_adaptive_test_step_phase1_start():
  """Tests Phase 1 initial descent proposal."""
  std_freqs = [1000.0, 2000.0]
  stimulus, _ = adaptive_audiometry.run_adaptive_test_step(
      past_results=[], standard_freqs=std_freqs, verbosity=3
  )
  # Default target is 1000 Hz, start level 60, descent 10
  # proposes 1000 at 50 if start was last heard
  # If NO target_freq_not_heard and NO target_freq_heard,
  # it uses start_level (60) - 10 = 50.
  assert stimulus == (1000.0, 50.0)


def test_run_adaptive_test_step_phase1_descent():
  """Tests Phase 1 continuing descent."""
  std_freqs = [1000.0, 2000.0]
  past_results = [(1000.0, 60.0, True)]
  stimulus, _ = adaptive_audiometry.run_adaptive_test_step(
      past_results=past_results, standard_freqs=std_freqs, verbosity=3
  )
  assert stimulus == (1000.0, 50.0)


def test_run_adaptive_test_step_phase1_stuck():
  """Tests transitioning out of Phase 1 when stuck at min level."""
  std_freqs = [1000.0, 2000.0]
  config = {'min_level_dbhl': 10.0}
  past_results = [(1000.0, 20.0, True), (1000.0, 10.0, True)]
  # Should move to Phase 2 (Sweep) because min level 10 was already heard
  # and it proposes 10 again.
  stimulus, _ = adaptive_audiometry.run_adaptive_test_step(
      past_results=past_results,
      standard_freqs=std_freqs,
      initial_phase_config=config,
      verbosity=3,
  )
  assert stimulus[0] == 2000.0
  assert stimulus[1] == 10.0


def test_run_adaptive_test_step_phase2_transition():
  """Tests transitioning to Phase 2 (Sweep) upon not hearing Phase 1."""
  std_freqs = [1000.0, 2000.0]
  past_results = [(1000.0, 60.0, True), (1000.0, 50.0, False)]
  stimulus, _ = adaptive_audiometry.run_adaptive_test_step(
      past_results=past_results, standard_freqs=std_freqs, verbosity=3
  )
  assert stimulus[0] == 2000.0
  assert stimulus[1] == 60.0  # From the 1000Hz threshold


def test_run_adaptive_test_step_phase2_sweep():
  """Tests Phase 2 stepping logic."""
  std_freqs = [1000.0, 2000.0, 4000.0]
  past_results = [
      (1000.0, 60.0, True),
      (1000.0, 50.0, False),
      (2000.0, 60.0, True),
  ]
  stimulus, _ = adaptive_audiometry.run_adaptive_test_step(
      past_results=past_results, standard_freqs=std_freqs, verbosity=3
  )
  assert stimulus[0] == 4000.0
  assert stimulus[1] == 45.0  # 60 - 15 (sweep step)


def test_run_adaptive_test_step_phase3_adaptive():
  """Tests Phase 3 triggering the kernel smoother."""
  std_freqs = [1000.0, 2000.0]
  # To trigger Phase 3, all standard freqs must have been tested,
  # AND target freq not heard at least once.
  past_results = [
      (1000.0, 60.0, True),
      (1000.0, 50.0, False),
      (2000.0, 60.0, True),
  ]
  stimulus, audiogram = adaptive_audiometry.run_adaptive_test_step(
      past_results=past_results, standard_freqs=std_freqs, verbosity=3
  )
  # Kernel smoother should give a prediction
  assert len(audiogram) == 2
  assert stimulus is not None


def test_create_test_grid_single_step():
  """Tests edge case where min and max frequency are the same."""
  config = adaptive_audiometry.GRID_CONFIG.copy()
  config['min_freq_hz'] = 1000.0
  config['max_freq_hz'] = 1000.0
  grid = adaptive_audiometry.create_test_grid(config)
  assert grid.shape[0] > 0


def test_run_adaptive_test_step_kernel_config():
  """Tests providing a custom kernel config."""
  std_freqs = [1000.0, 2000.0]
  stimulus, _ = adaptive_audiometry.run_adaptive_test_step(
      past_results=[],
      standard_freqs=std_freqs,
      kernel_config={'epsilon': 1e-5},
      verbosity=3,
  )
  assert stimulus is not None


def test_run_adaptive_test_step_phase2_no_target_heard():
  """Tests Phase 2 fallback when the target frequency was never heard."""
  std_freqs = [1000.0, 2000.0]
  past_results = [(1000.0, 60.0, False)]
  stimulus, _ = adaptive_audiometry.run_adaptive_test_step(
      past_results=past_results, standard_freqs=std_freqs, verbosity=3
  )
  assert stimulus[1] == 60.0
