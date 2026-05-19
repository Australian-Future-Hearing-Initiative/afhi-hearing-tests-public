"""Tests for calibration.py."""

import numpy as np
import unittest
import streamlit as st
from unittest.mock import patch

import calibration
import common

class TestCalibration(unittest.TestCase):
  """Test cases for calibration.py functions."""

  def setUp(self):
    """Set up test fixtures."""
    # Mock streamlit session state.
    if not hasattr(st, 'session_state'):
      st.session_state = {}

  def test_play_calibration_tone_raises_error_for_high_volume(self):
    """Test that play_calibration_tone raises error for volume > 1."""
    with self.assertRaises(ValueError):
      calibration.play_calibration_tone(1000, 1.1)

  def test_play_calibration_tone_raises_error_for_negative_volume(self):
    """Test that play_calibration_tone raises error for volume < 0."""
    with self.assertRaises(ValueError):
      calibration.play_calibration_tone(1000, -0.1)

  @patch('calibration.common.autoplay_audio')
  @patch('calibration.wavfile.write')
  def test_play_calibration_tone_uses_correct_sampling_rate(self,
                                                            mock_write, _):
    """Test that play_calibration_tone uses the correct sampling rate."""
    calibration.play_calibration_tone(1000, 0.5, 0.1)
    _, fs, _ = mock_write.call_args[0]
    self.assertEqual(fs, calibration.FS_HZ)

  @patch('calibration.common.autoplay_audio')
  @patch('calibration.wavfile.write')
  def test_play_calibration_tone_generates_correct_data_format(self,
                                                               mock_write, _):
    """Test that play_calibration_tone generates data with correct format."""
    duration = 0.1
    calibration.play_calibration_tone(1000, 0.5, duration)
    _, _, data = mock_write.call_args[0]
    self.assertIsInstance(data, np.ndarray)
    self.assertEqual(data.dtype, np.int16)
    self.assertEqual(len(data), int(calibration.FS_HZ * duration))

  @patch('calibration.common.autoplay_audio')
  @patch('calibration.wavfile.write')
  def test_play_calibration_tone_scales_amplitude_correctly(self,
                                                            mock_write, _):
    """Test that play_calibration_tone scales the amplitude correctly."""
    volume = 0.5
    calibration.play_calibration_tone(1000, volume, 0.1)
    _, _, data = mock_write.call_args[0]
    max_amplitude = np.max(np.abs(data))
    expected_max = volume * common.MAX_16_BIT_INT
    np.testing.assert_allclose(max_amplitude, expected_max, rtol=0.01)

  @patch('calibration.play_calibration_tone')
  def test_calibrate_audio_plays_tone_on_first_call(self, mock_play):
    """Test that calibrate_audio plays tone when first called."""
    st.session_state.calibration_playing = False
    calibration.calibrate_audio()
    mock_play.assert_called_once_with(
      calibration.CALIBRATION_TONE_FREQ,
      calibration.REFERENCE_AMPLITUDE,
      calibration.CALIBRATION_TONE_DURATION
    )
    self.assertFalse(st.session_state.calibration_playing)

  @patch('calibration.play_calibration_tone')
  def test_calibrate_audio_stops_on_second_call(self, mock_play):
    """Test that calibrate_audio stops playing when called while playing."""
    st.session_state.calibration_playing = True
    calibration.calibrate_audio()
    self.assertFalse(st.session_state.calibration_playing)
    self.assertEqual(mock_play.call_count, 0)

  def test_dbhl_to_dbspl_converts_correctly_for_pixel_buds(self):
    """Test that dbhl_to_dbspl converts values correctly for Pixel Buds."""
    config = 'PixelBuds_HughsonWestlake'
    self.assertEqual(calibration.dbhl_to_dbspl(0, 1000, config), 5.7)
    self.assertEqual(calibration.dbhl_to_dbspl(50, 2000, config), 56.2)

  def test_dbhl_to_dbspl_raises_error_for_unknown_hardware(self):
    """Test that dbhl_to_dbspl raises error for unknown hardware."""
    with self.assertRaises(ValueError):
      calibration.dbhl_to_dbspl(0, 1000, 'Unknown Hardware')

  def test_dbhl_to_dbspl_raises_error_for_airpods(self):
    """Test that dbhl_to_dbspl raises error for unsupported AirPods."""
    with self.assertRaises(ValueError):
      calibration.dbhl_to_dbspl(0, 1000, 'AirPods')

  def test_dbspl_to_amp_converts_reference_point_correctly(self):
    """Test that dbspl_to_amp converts the reference point correctly."""
    amp = calibration.dbspl_to_amp(
      calibration.REFERENCE_DB_SPL,
      calibration.REFERENCE_DB_SPL,
      calibration.REFERENCE_AMPLITUDE
    )
    self.assertEqual(amp, calibration.REFERENCE_AMPLITUDE)

  def test_dbspl_to_amp_handles_20db_increase(self):
    """Test that dbspl_to_amp correctly handles a 20dB increase."""
    amp = calibration.dbspl_to_amp(
      calibration.REFERENCE_DB_SPL + 20,
      calibration.REFERENCE_DB_SPL,
      calibration.REFERENCE_AMPLITUDE
    )
    np.testing.assert_allclose(amp, calibration.REFERENCE_AMPLITUDE * 10)

  def test_dbspl_to_amp_handles_20db_decrease(self):
    """Test that dbspl_to_amp correctly handles a 20dB decrease."""
    amp = calibration.dbspl_to_amp(
      calibration.REFERENCE_DB_SPL - 20,
      calibration.REFERENCE_DB_SPL,
      calibration.REFERENCE_AMPLITUDE
    )
    np.testing.assert_allclose(amp, calibration.REFERENCE_AMPLITUDE * 0.1)

  def test_dbspl_to_amp_works_with_different_reference_values(self):
    """Test that dbspl_to_amp works correctly with different reference."""
    amp = calibration.dbspl_to_amp(80, 60, 0.1)
    np.testing.assert_allclose(amp, 1.0)

  def test_amp_to_dbspl_converts_reference_point_correctly(self):
    """Test that amp_to_dbspl converts the reference point correctly."""
    db_spl = calibration.amp_to_dbspl(
      calibration.REFERENCE_AMPLITUDE,
      calibration.REFERENCE_DB_SPL,
      calibration.REFERENCE_AMPLITUDE
    )
    self.assertEqual(db_spl, calibration.REFERENCE_DB_SPL)

  def test_amp_to_dbspl_handles_10x_amplitude(self):
    """Test that amp_to_dbspl correctly handles 10x amplitude."""
    db_spl = calibration.amp_to_dbspl(
      calibration.REFERENCE_AMPLITUDE * 10,
      calibration.REFERENCE_DB_SPL,
      calibration.REFERENCE_AMPLITUDE
    )
    np.testing.assert_allclose(db_spl, calibration.REFERENCE_DB_SPL + 20)

  def test_amp_to_dbspl_handles_0_1x_amplitude(self):
    """Test that amp_to_dbspl correctly handles 0.1x amplitude."""
    db_spl = calibration.amp_to_dbspl(
      calibration.REFERENCE_AMPLITUDE * 0.1,
      calibration.REFERENCE_DB_SPL,
      calibration.REFERENCE_AMPLITUDE
    )
    np.testing.assert_allclose(db_spl, calibration.REFERENCE_DB_SPL - 20)

  def test_amp_to_dbspl_works_with_different_reference_values(self):
    """Test that amp_to_dbspl works with different reference values."""
    db_spl = calibration.amp_to_dbspl(1.0, 60, 0.1)
    np.testing.assert_allclose(db_spl, 80)  # 10x amplitude = 20dB increase

  def test_amp_to_dbspl_and_dbspl_to_amp_are_inverse_functions(self):
    """Test that amp_to_dbspl and dbspl_to_amp are inverses of each other."""
    # Test a range of amplitudes
    test_amplitudes = [0.001, 0.01, 0.1, 1.0]
    for amp in test_amplitudes:
      db_spl = calibration.amp_to_dbspl(amp)
      amp_converted_back = calibration.dbspl_to_amp(db_spl)
      np.testing.assert_allclose(amp, amp_converted_back)

if __name__ == '__main__':
  unittest.main()
