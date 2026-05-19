"""Unit tests for audio_synthesis.py."""

# pylint: disable=protected-access

import numpy as np
import pytest
from unittest.mock import patch

import audio_synthesis


class TestAudioSynthesis:
  """Tests for functions in audio_synthesis.py."""

  def test_calculate_rms(self):
    """Tests RMS calculation on mathematical signals."""
    # DC signal
    assert np.isclose(
        audio_synthesis._calculate_rms(np.array([1.0, 1.0, 1.0])), 1.0
    )
    # AC signal
    assert np.isclose(
        audio_synthesis._calculate_rms(np.array([1.0, -1.0, 1.0, -1.0])), 1.0
    )
    # Zero signal
    assert np.isclose(audio_synthesis._calculate_rms(np.zeros(10)), 0.0)

  def test_basic_speech_shaped_noise(self):
    """Tests that the basic speech shaped noise filter runs correctly."""
    white_noise = np.random.randn(44100)
    shaped = audio_synthesis.basic_speech_shaped_noise(white_noise, 44100)
    assert len(shaped) == len(white_noise)
    assert shaped.shape == white_noise.shape
    # Ensure it actually modified the noise
    assert not np.allclose(shaped, white_noise)

  def test_advanced_speech_shaped_noise_valid(self):
    """Tests the advanced speech shaped noise filter on a valid sample rate."""
    white_noise = np.random.randn(44100)
    shaped = audio_synthesis.advanced_speech_shaped_noise(white_noise, 44100)
    assert len(shaped) == len(white_noise)
    # Check normalization to peak amplitude of 1.0
    assert np.max(np.abs(shaped)) <= 1.0
    assert not np.allclose(shaped, white_noise)

  def test_advanced_speech_shaped_noise_invalid_sr(self):
    """Tests that advanced noise shaping rejects too-low sample rates."""
    with pytest.raises(
        ValueError, match='Sample rate must be at least 32000 Hz'
    ):
      audio_synthesis.advanced_speech_shaped_noise(np.random.randn(100), 16000)

  def test_mix_vcv_with_noise_routing(self):
    """Tests that the 'ear' parameter correctly spatializes the audio."""
    signal = np.ones(1000) * 0.1
    with patch('common.get_active_signal_rms', return_value=0.1):
      # Ear = left
      mixed_left, _ = audio_synthesis.mix_vcv_with_noise(
          snr_db=10, ear='left', audio_data=signal, sample_rate=44100
      )
      assert mixed_left.shape == (1000, 2)
      assert np.all(mixed_left[:, 1] == 0)
      assert not np.all(mixed_left[:, 0] == 0)

      # Ear = right
      mixed_right, _ = audio_synthesis.mix_vcv_with_noise(
          snr_db=10, ear='right', audio_data=signal, sample_rate=44100
      )
      assert np.all(mixed_right[:, 0] == 0)
      assert not np.all(mixed_right[:, 1] == 0)

      # Ear = both
      mixed_both, _ = audio_synthesis.mix_vcv_with_noise(
          snr_db=10, ear='both', audio_data=signal, sample_rate=44100
      )
      assert np.all(mixed_both[:, 0] == mixed_both[:, 1])

  @patch('common.get_active_signal_rms', return_value=0.1)
  @patch('numpy.random.randn')
  def test_mix_vcv_with_noise_snr_math(self, mock_randn, _):
    """Tests the exact mathematical scaling of noise relative to signal."""
    signal = np.ones(100) * 0.1
    # Force the random noise to be exactly 1.0 everywhere.
    mock_randn.return_value = np.ones(100)

    # snr = 0 means signal_rms == noise_rms
    mixed, _ = audio_synthesis.mix_vcv_with_noise(
        snr_db=0,
        audio_data=signal,
        sample_rate=44100,
        noise_type='White Noise',
        ear='both',
    )
    # The output is signal (0.1) + noise (scaled to 0.1) = 0.2 everywhere.
    assert np.allclose(mixed, 0.2)

  def test_mix_vcv_with_noise_datatypes(self):
    """Tests that integer and float datatypes are preserved correctly."""
    with patch('common.get_active_signal_rms', return_value=0.1):
      # Test float64
      signal_float = np.ones(100, dtype=np.float64) * 0.1
      mixed_float, _ = audio_synthesis.mix_vcv_with_noise(
          snr_db=0, audio_data=signal_float, sample_rate=44100
      )
      assert mixed_float.dtype == np.float64

      # Test int16
      signal_int = np.ones(100, dtype=np.int16) * 3000
      mixed_int, _ = audio_synthesis.mix_vcv_with_noise(
          snr_db=0, audio_data=signal_int, sample_rate=44100
      )
      assert mixed_int.dtype == np.int16

  def test_mix_vcv_with_noise_target_rms(self):
    """Tests the global output normalization argument."""
    signal = np.ones(1000) * 0.1
    target = 0.1
    with patch('common.get_active_signal_rms', return_value=0.1):
      mixed, _ = audio_synthesis.mix_vcv_with_noise(
          snr_db=0,
          audio_data=signal,
          sample_rate=44100,
          target_output_rms=target,
          ear='left',
      )
    # The left channel RMS should be exactly the target.
    assert np.isclose(audio_synthesis._calculate_rms(mixed[:, 0]), target)

  def test_mix_vcv_with_noise_clipping_warning(self):
    """Tests that extreme noise levels trigger a warning and are clipped."""
    signal = np.ones(1000) * 0.5
    with patch('common.get_active_signal_rms', return_value=0.5):
      with pytest.warns(UserWarning, match='Clipping occurred'):
        mixed, _ = audio_synthesis.mix_vcv_with_noise(
            snr_db=-100, audio_data=signal, sample_rate=44100
        )
    # It should be clipped to 1.0
    assert np.max(np.abs(mixed)) <= 1.0

  def test_mix_vcv_with_noise_silent_signal(self):
    """Tests that completely silent signals trigger a warning and bypass."""
    signal = np.zeros(1000)
    with pytest.warns(UserWarning, match='Input signal is silent'):
      mixed, _ = audio_synthesis.mix_vcv_with_noise(
          snr_db=0, audio_data=signal, sample_rate=44100
      )
    assert np.all(mixed == 0)

  def test_mix_vcv_with_noise_ramp(self):
    """Tests the half-cosine ramping on the noise envelope."""
    signal = np.zeros(10000)
    # We must patch get_active_signal_rms to not return 0, otherwise it aborts.
    with patch('common.get_active_signal_rms', return_value=0.1):
      mixed, _ = audio_synthesis.mix_vcv_with_noise(
          snr_db=0, audio_data=signal, sample_rate=44100, ramp_duration_s=0.1
      )
    # Check that it ramped. A ramp means the start and end are 0.
    assert mixed[0, 0] == 0.0
    assert mixed[-1, 0] == 0.0

  def test_mix_vcv_with_noise_requires_data(self):
    """Tests the validation on input data arguments."""
    with pytest.raises(
        ValueError, match='Must provide either audio_data or clean_vcv_path'
    ):
      audio_synthesis.mix_vcv_with_noise(snr_db=0)

  @patch('scipy.io.wavfile.read')
  def test_mix_vcv_with_noise_reads_path(self, mock_read):
    """Tests that providing a file path correctly invokes wavfile.read."""
    mock_read.return_value = (44100, np.ones(100) * 0.1)
    with patch('common.get_active_signal_rms', return_value=0.1):
      _, sr = audio_synthesis.mix_vcv_with_noise(
          snr_db=0, clean_vcv_path='dummy.wav'
      )
    assert sr == 44100
    mock_read.assert_called_once_with('dummy.wav')

  @patch(
      'audio_synthesis.basic_speech_shaped_noise',
      return_value=np.ones(100) * 0.1,
  )
  def test_mix_vcv_with_noise_calls_basic(self, mock_basic):
    """Tests that noise types invoke the proper sub-functions."""
    signal = np.ones(100) * 0.1
    with patch('common.get_active_signal_rms', return_value=0.1):
      audio_synthesis.mix_vcv_with_noise(
          snr_db=0,
          audio_data=signal,
          sample_rate=44100,
          noise_type='Basic Speech-Shaped Noise',
      )
    mock_basic.assert_called_once()

  @patch(
      'audio_synthesis.advanced_speech_shaped_noise',
      return_value=np.ones(100) * 0.1,
  )
  def test_mix_vcv_with_noise_calls_advanced(self, mock_advanced):
    """Tests that noise types invoke the proper sub-functions."""
    signal = np.ones(100) * 0.1
    with patch('common.get_active_signal_rms', return_value=0.1):
      audio_synthesis.mix_vcv_with_noise(
          snr_db=0,
          audio_data=signal,
          sample_rate=44100,
          noise_type='Advanced Speech-Shaped Noise',
      )
    mock_advanced.assert_called_once()

  def test_mix_vcv_with_noise_silent_noise(self):
    """Tests division-by-zero protection for silent noise generation."""
    signal = np.ones(100) * 0.1
    with patch('common.get_active_signal_rms', return_value=0.1):
      with patch('numpy.random.randn', return_value=np.zeros(100)):
        mixed, _ = audio_synthesis.mix_vcv_with_noise(
            snr_db=0, audio_data=signal, sample_rate=44100
        )
    # With silent noise, the output should just be the signal itself
    assert np.allclose(mixed[:, 0], 0.1)
