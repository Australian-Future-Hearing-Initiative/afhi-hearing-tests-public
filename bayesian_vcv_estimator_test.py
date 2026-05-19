"""Unit tests for bayesian_vcv_estimator.py."""

import numpy as np

from bayesian_vcv_estimator import ZestEstimator


def test_initialization():
  """Tests initialization of the ZestEstimator."""
  estimator = ZestEstimator(prior_mean=10.0, prior_sd=5.0)
  assert not estimator.history

  posterior = np.exp(estimator.log_posterior)
  # Check it sums to 1.0 (valid PDF)
  assert np.isclose(np.sum(posterior), 1.0)

  # Check mean is close to 10.0
  mean_val = np.sum(posterior * estimator.grid)
  assert np.isclose(mean_val, 10.0, atol=0.1)


def test_reinitialize_prior_empty_history():
  """Tests reinitialization when history is empty."""
  estimator = ZestEstimator(prior_mean=10.0, prior_sd=5.0)
  estimator.reinitialize_prior(prior_mean=20.0, prior_sd=2.0)

  posterior = np.exp(estimator.log_posterior)
  mean_val = np.sum(posterior * estimator.grid)
  assert np.isclose(mean_val, 20.0, atol=0.1)


def test_reinitialize_prior_active_history():
  """Tests reinitialization is blocked when history is active."""
  estimator = ZestEstimator(prior_mean=10.0, prior_sd=5.0)
  estimator.history.append((5.0, True))

  # Attempt to reinitialize
  estimator.reinitialize_prior(prior_mean=20.0, prior_sd=2.0)

  # Should remain around 10.0, not 20.0
  posterior = np.exp(estimator.log_posterior)
  mean_val = np.sum(posterior * estimator.grid)
  assert np.isclose(mean_val, 10.0, atol=0.1)


def test_psychometric_function():
  """Tests the psychometric function at key limits."""
  snr = 0.0
  threshold = 0.0
  slope = 1.0
  chance_rate = 0.1
  lapse_rate = 0.02

  # At threshold, it should be exactly halfway between chance and 1-lapse
  p_at_threshold = ZestEstimator.psychometric_function(
      snr, threshold, slope, chance_rate, lapse_rate
  )
  expected_midpoint = chance_rate + (1.0 - chance_rate - lapse_rate) * 0.5
  assert np.isclose(p_at_threshold, expected_midpoint)

  # At high SNR (very easy), it should approach 1 - lapse
  p_high_snr = ZestEstimator.psychometric_function(
      100.0, threshold, slope, chance_rate, lapse_rate
  )
  assert np.isclose(p_high_snr, 1.0 - lapse_rate)

  # At low SNR (impossible), it should approach chance
  p_low_snr = ZestEstimator.psychometric_function(
      -100.0, threshold, slope, chance_rate, lapse_rate
  )
  assert np.isclose(p_low_snr, chance_rate)


def test_get_next_snr():
  """Tests that get_next_snr returns the mean of the posterior."""
  # Use a small prior_sd to avoid truncation bias from the grid bounds [-55, 40]
  estimator = ZestEstimator(prior_mean=15.0, prior_sd=2.0)
  next_snr = estimator.get_next_snr()
  assert np.isclose(next_snr, 15.0, atol=0.1)


def test_update_correct_response():
  """Tests that a correct response lowers the estimated threshold."""
  estimator = ZestEstimator(prior_mean=10.0)
  initial_mean, _ = estimator.get_estimate()

  estimator.update(snr=10.0, is_correct=True)
  new_mean, _ = estimator.get_estimate()

  assert new_mean < initial_mean
  assert len(estimator.history) == 1


def test_update_incorrect_response():
  """Tests that an incorrect response raises the estimated threshold."""
  estimator = ZestEstimator(prior_mean=10.0)
  initial_mean, _ = estimator.get_estimate()

  estimator.update(snr=10.0, is_correct=False)
  new_mean, _ = estimator.get_estimate()

  assert new_mean > initial_mean
  assert len(estimator.history) == 1


def test_get_estimate_uncertainty_shrinks():
  """Tests that repeated updates reduce standard deviation (uncertainty)."""
  estimator = ZestEstimator(prior_mean=10.0, prior_sd=20.0)
  _, initial_sd = estimator.get_estimate()

  # Simulate a few trials around threshold
  estimator.update(snr=10.0, is_correct=True)
  estimator.update(snr=5.0, is_correct=False)
  estimator.update(snr=8.0, is_correct=True)

  _, new_sd = estimator.get_estimate()
  assert new_sd < initial_sd
