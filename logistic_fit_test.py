"""Tests for logistic_fit.py."""

import unittest
from unittest.mock import patch
import logistic_fit

class TestLogisticFunctions(unittest.TestCase):
  """Tests for the core logistic math functions."""

  def test_psychometric_logistic_curve_at_threshold(self):
    """At threshold, probability should be 0.5."""
    level = 20.0
    threshold = 20.0
    spread = 5.0
    prob = logistic_fit.psychometric_logistic_curve(level, threshold, spread)
    self.assertAlmostEqual(prob, 0.5)

  def test_psychometric_logistic_curve_limits(self):
    """Test limits far above and below threshold."""
    threshold = 50.0
    spread = 2.0 # Steep curve.
    # Case 1: Level = 80 dBHL (much louder than 50).
    p_loud = logistic_fit.psychometric_logistic_curve(80.0, threshold, spread)
    self.assertAlmostEqual(p_loud, 1.0, places=4)
    # Case 2: Level = 20 dBHL (much softer than 50).
    p_soft = logistic_fit.psychometric_logistic_curve(20.0, threshold, spread)
    self.assertAlmostEqual(p_soft, 0.0, places=4)


class TestLocalGridFitter(unittest.TestCase):
  """Tests for the LocalGridFitter class."""

  def setUp(self):
    self.fitter = logistic_fit.LocalGridFitter()

  def test_fit_perfect_step(self):
    """Test fitting a perfect step function response."""
    # Responses transition from False (0) to True (1) at 30dB.
    # 10, 20 dB -> Not Heard
    # 40, 50 dB -> Heard
    levels = [10.0, 20.0, 40.0, 50.0]
    responses = [0.0, 0.0, 1.0, 1.0] # Using float 0/1 as expected by fitter
    threshold, spread = self.fitter.fit(levels, responses)
    # Threshold should be roughly between 20 and 40. ideally 30.
    self.assertTrue(20.0 <= threshold <= 40.0,
                    f'Threshold {threshold} not in expected range [20, 40]')
    # Spread should be relatively small (steep) for perfect data
    self.assertLess(spread, 5.0)

  def test_fit_noisy_data(self):
    """Test fitting with some noise."""
    # 30dB was heard, 20dB not heard.
    levels = [10, 20, 30, 40, 50]
    responses = [0.0, 0.0, 1.0, 1.0, 1.0]
    threshold, _ = self.fitter.fit(levels, responses)
    # Threshold should be around 25 (midpoint of 20 and 30)
    self.assertAlmostEqual(threshold, 25.0, delta=5.0)


class TestGlobalDictionaryFitter(unittest.TestCase):
  """Tests for the GlobalDictionaryFitter."""

  def test_fit_returns_dict(self):
    """Test that fit returns a dictionary with keys for all frequencies."""
    freqs = [250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
    fitter = logistic_fit.GlobalDictionaryFitter(freqs)
    # Some dummy data.
    results = [
        (1000.0, 50.0, True),
        (1000.0, 30.0, False),
        (4000.0, 60.0, True),
        (4000.0, 40.0, False)
    ]
    audiogram = fitter.fit(results)
    self.assertIsInstance(audiogram, dict)
    self.assertEqual(len(audiogram), len(freqs))
    for f in freqs:
      self.assertIn(f, audiogram)
      self.assertIsNotNone(audiogram[f])

  def test_fit_no_data_raises_error(self):
    """Test that empty history raises ValueError."""
    freqs = [1000.0]
    fitter = logistic_fit.GlobalDictionaryFitter(freqs)
    with self.assertRaisesRegex(ValueError,
                                'Cannot fit audiogram with empty results'):
      fitter.fit([])

  def test_global_fitter_cache_invalidation(self):
    """Test that shape cache invalidates when frequencies change."""
    # 1. Initialize with single frequency
    freqs1 = [1000.0]
    fitter1 = logistic_fit.GlobalDictionaryFitter(freqs1)
    shapes1 = fitter1.shapes
    self.assertEqual(shapes1.shape[1], 1)
    # 2. Initialize with two frequencies.
    freqs2 = [1000.0, 2000.0]
    fitter2 = logistic_fit.GlobalDictionaryFitter(freqs2)
    shapes2 = fitter2.shapes
    self.assertEqual(shapes2.shape[1], 2)
    # Verify they are different objects or at least different sizes.
    self.assertNotEqual(shapes1.shape, shapes2.shape)


class TestIntegration(unittest.TestCase):
  """Integration tests for the top-level functions."""

  def test_fit_audiogram_hybrid_structure(self):
    """Test that the hybrid fit returns the expected structure."""
    freqs = [500.0, 1000.0, 2000.0]
    results = [
        (1000.0, 40.0, True),
        (1000.0, 20.0, False),
        (500.0, 45.0, True),
        (500.0, 25.0, False),
        (2000.0, 50.0, True),
        (2000.0, 30.0, False)
    ]
    hybrid, local, global_res = logistic_fit.fit_audiogram_hybrid(results,
                                                                  freqs)
    # Check Hybrid structure.
    self.assertIn('thresholds', hybrid)
    self.assertIn('variances', hybrid)
    self.assertEqual(len(hybrid['thresholds']), 3)
    # Check Local structure.
    self.assertIn('thresholds', local)
    self.assertIn('spreads', local)
    self.assertIn('variances', local)
    # Check Global structure.
    self.assertIsInstance(global_res, dict)
    self.assertEqual(len(global_res), 3)
    # Basic value sanity check.
    t_1k = hybrid['thresholds'][1000.0]
    self.assertTrue(20.0 < t_1k < 40.0,
                    f'Estimated 1k threshold {t_1k} outside expected range')

  def test_fit_audiogram_logistic_no_data_raises_error(self):
    """Test that logistic fit raises ValueError with empty data."""
    standard_freqs = [1000.0]
    with self.assertRaisesRegex(ValueError,
                                'Cannot fit audiogram with empty results'):
      logistic_fit.fit_audiogram_logistic([], standard_freqs)

  def test_fit_audiogram_logistic_missing_frequency(self):
    """Test fit_audiogram_logistic with missing data for some frequencies."""
    standard_freqs = [1000.0, 2000.0]
    # Only provide data for 1000 Hz
    past_results = [
        (1000.0, 50.0, True),
        (1000.0, 10.0, False)
    ]
    est_audiogram, spreads, variances = logistic_fit.fit_audiogram_logistic(
        past_results, standard_freqs
    )
    # 1000 Hz should have results.
    self.assertIsNotNone(est_audiogram[1000.0])
    self.assertIsNotNone(spreads[1000.0])
    self.assertIsNotNone(variances[1000.0])
    # 2000 Hz should be None.
    self.assertIsNone(est_audiogram[2000.0])
    self.assertIsNone(spreads[2000.0])
    self.assertIsNone(variances[2000.0])

  def test_fit_audiogram_hybrid_missing_local(self):
    """Test hybrid fit handles cases where local fit is None."""
    standard_freqs = [1000.0, 2000.0]
    # Only provide data for 1000 Hz.
    past_results = [
        (1000.0, 50.0, True),
        (1000.0, 10.0, False)
    ]
    hybrid, local, global_res = logistic_fit.fit_audiogram_hybrid(
        past_results, standard_freqs
    )
    # Global fit should interpolate/guess for 2000 Hz.
    self.assertIsNotNone(global_res[2000.0])
    # Local fit should be None for 2000 Hz.
    self.assertIsNone(local['thresholds'][2000.0])
    # Hybrid fit should currently return None if local is missing
    # (as per implementation logic: if t_local is None... continue)
    self.assertIsNone(hybrid['thresholds'][2000.0])
    self.assertIsNone(hybrid['variances'][2000.0])

  @patch('logistic_fit._LOCAL_FITTER.fit')
  def test_label_smoothing_applied(self, mock_fit):
    """Test that boolean responses are smoothed before fitting."""
    # Setup mock to return dummy values
    mock_fit.return_value = (30.0, 5.0)
    standard_freqs = [1000.0]
    past_results = [
        (1000.0, 50.0, True),   # Should become 1.0 - LAPSE_RATE
        (1000.0, 10.0, False)   # Should become LAPSE_RATE
    ]
    logistic_fit.fit_audiogram_logistic(past_results, standard_freqs)
    # Get arguments passed to fit.
    call_args = mock_fit.call_args
    # args[0] is levels, args[1] is responses.
    fit_responses = call_args[0][1]
    # Check values. Note: the fit adds 2 priors at the end, so check first 2.
    lapse_rate = logistic_fit.LAPSE_RATE
    self.assertAlmostEqual(fit_responses[0], 1.0 - lapse_rate)
    self.assertAlmostEqual(fit_responses[1], lapse_rate)

if __name__ == '__main__':
  unittest.main()
