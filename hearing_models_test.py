"""Tests for the hearing_models module."""

import numpy as np
import unittest

import hearing_models

class TestHearingModels(unittest.TestCase):
  """Tests for all functions in hearing_models."""

  def test_phons_to_dbspl(self):
    """Tests the phons_to_dbspl function."""
    freq_hz = np.array([1000.0, 2000.0])
    phon_level = np.array([40.0, 60.0])
    actual = hearing_models.phons_to_dbspl(freq_hz, phon_level)
    expected = np.array([[40.9245], [60.367467]])
    np.testing.assert_allclose(actual, expected)

  def test_dbhl_to_slopes(self):
    """Tests the dbhl_to_slopes function."""
    freq_hz = np.array([1000.0, 2000.0, 4000.0])
    dbhl = np.array([20.0, 30.0, 40.0])
    actual = hearing_models.dbhl_to_slopes(freq_hz, dbhl)
    expected = np.array([[1.301248, 1.532005, 1.862246],
                        [1.289089, 1.506904, 1.813293],
                        [1.285166, 1.498882, 1.797854]])
    np.testing.assert_allclose(actual, expected, atol=1e-6)

  def test_hearing_level_model_raises_error(self):
    """Tests that hearing_level_model raises an error for invalid inputs."""
    too_few_model_coeffs = np.array([])
    too_many_model_coeffs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    with self.assertRaises(ValueError):
      hearing_models.hearing_level_model(np.array([1000.0]),
                                         too_few_model_coeffs)
    with self.assertRaises(ValueError):
      hearing_models.hearing_level_model(np.array([1000.0]),
                                         too_many_model_coeffs)

  def test_hearing_level_model_with_normal_hearing(self):
    """Tests hearing_level_model with normal hearing."""
    freqs_hz = np.array([1000.0, 2000.0, 4000.0])
    model_coeffs = np.array([0.0])
    actual_dbspl, actual_slopes = hearing_models.hearing_level_model(
        freqs_hz, model_coeffs)
    expected_dbspl = np.array([0, 0, 0]).reshape(-1, 1)
    expected_slopes = np.array([1, 1, 1]).reshape(-1, 1)
    np.testing.assert_allclose(actual_dbspl, expected_dbspl)
    np.testing.assert_allclose(actual_slopes, expected_slopes)

  def test_sones_subject_to_sones_nh(self):
    """Tests the sones_subject_to_sones_nh function."""
    # If the model_coeffs are [0], then the output should be the same as the
    # input.
    sones = np.array([1.0, 20.0, 30.0])
    freqs_hz = np.array([1000.0, 2000.0, 4000.0])
    model = {'component_coeffs': np.array([0.0]), 'sone_intersection': 24.0}
    sones_nh = hearing_models.sones_subject_to_sones_nh(sones, freqs_hz, model)
    self.assertTrue(np.allclose(sones_nh, sones))

  def test_random_test_frequencies_and_amplitudes(self):
    """Tests that random numbers are generated with the correct dimensions."""
    min_freq = 1000.0
    max_freq = 6000.0
    n_samples = 10
    loudness_model = {'component_coeffs': np.array([1.0, 0.0]),
                      'sone_intersection': 24.0}
    reqs, amp = hearing_models.random_test_frequencies_and_amplitudes(
      min_freq, max_freq, n_samples, loudness_model)
    self.assertEqual(reqs.shape, (n_samples, 1))
    self.assertEqual(amp.shape, (n_samples, 1))

  def test_hearing_level_model_to_audiogram(self):
    """Tests the hearing_level_model_to_audiogram function."""
    model = {'component_coeffs': np.asarray([1.0, 0.0]),
             'sone_intersection': 24.0}
    actual = hearing_models.loudness_model_to_audiogram(model)
    expected_freqs = np.array(
      [250, 500, 1000, 1500, 2000, 3000, 4000, 6000, 8000]).reshape(-1, 1)
    np.testing.assert_allclose(actual['frequencies'], expected_freqs)
    # TODO: Add more tests for the amplitude values.

  def test_simulate_loudness_categorization(self):
    """Tests the simulate_loudness_categorization function."""
    freq_hz = 1000.0
    amp = 0.5
    model = {'component_coeffs': np.asarray([1.0, 0.0]),
             'sone_intersection': 24.0}
    actual = hearing_models.simulate_loudness_categorization(freq_hz, amp,
                                                             model, 0)
    expected = 25
    self.assertEqual(actual, expected)

  def test_update_loudness_model(self):
    """Tests the update_loudness_model function."""
    frequencies = np.array([1000.0, 2000.0, 4000.0]).reshape(-1, 1)
    amplitudes = np.array([0.5, 0.6, 0.7]).reshape(-1, 1)
    cus = np.array([30.0, 25.0, 20.0]).reshape(-1, 1)
    model = {'component_coeffs': np.asarray([1.0, 0.0, 0.0]),
             'sone_intersection': 24.0}
    rate = 0.5
    py_model, _ = hearing_models.update_loudness_model(
      frequencies, amplitudes, cus, model, rate)
    self.assertEqual(py_model['component_coeffs'].shape, (3,))
    # TODO: add a smarter test.

if __name__ == '__main__':
  unittest.main()
