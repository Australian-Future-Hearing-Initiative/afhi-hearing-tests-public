"""Unit tests for common.py."""

import io
import os
import tempfile
import unittest
import zipfile
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
import streamlit as st
from scipy.io import wavfile as _wavfile

import common

class TestCommon(unittest.TestCase):
  """Test class for common utility functions."""

  def setUp(self):
    # Mock streamlit session state.
    if not hasattr(st, 'session_state'):
      st.session_state = {}
    # Create a mock for the audio container with components.v1.html.
    mock_container = MagicMock()
    mock_container.components = MagicMock()
    mock_container.components.v1 = MagicMock()
    mock_container.components.v1.html = MagicMock()
    st.session_state.audio_container = mock_container
    # Create standard test data used by multiple tests.
    self.text_content = 'Test content'
    self.fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    # Use self.fig for matplotlib figures to ensure cleanup.
    self.test_files = [
        ('test.txt', self.text_content),
        ('test.png', self.fig)
    ]
    self.fig.clear()
    plt.close(self.fig)

  def test_generate_zip_bytes(self):
    """Tests the generate_zip_bytes function directly."""
    zip_data_bytes = common.generate_zip_bytes(self.test_files)
    self.assertIsInstance(zip_data_bytes, bytes)
    self.assertGreater(len(zip_data_bytes), 0)
    # Verify zip contents from bytes.
    zip_data_io = io.BytesIO(zip_data_bytes)
    with zipfile.ZipFile(zip_data_io, 'r') as zip_file:
      # Check if both files are in the zip.
      self.assertIn('test.txt', zip_file.namelist())
      self.assertIn('test.png', zip_file.namelist())
      # Verify text content.
      self.assertEqual(zip_file.read('test.txt').decode(), self.text_content)
      # Verify png file exists and is non-empty.
      self.assertGreater(zip_file.getinfo('test.png').file_size, 0)


if __name__ == '__main__':
  unittest.main()


# ---------------------------------------------------------------------------
# pytest-style tests for audio helpers
# ---------------------------------------------------------------------------

def _write_temp_wav(data, sample_rate=44100):
  '''Writes numpy data to a temporary WAV file and returns its path.'''
  handle, path = tempfile.mkstemp(suffix='.wav')
  os.close(handle)
  _wavfile.write(path, sample_rate, data)
  return path


# --- read_wav_as_float ---

def test_read_wav_as_float_int16_normalizes():
  '''int16 WAV is normalized to float32 with values in [-1, 1].'''
  data = np.array([0, 16383, -16384, 32767], dtype=np.int16)
  path = _write_temp_wav(data)
  try:
    result, fs, orig_dtype = common.read_wav_as_float(path)
    assert fs == 44100
    assert orig_dtype == np.dtype('int16')
    assert result.dtype == np.float32
    assert np.all(result >= -1.0) and np.all(result <= 1.0)
    # 32767 / 32767 == 1.0.
    assert result[-1] == pytest.approx(1.0, abs=1e-4)
  finally:
    os.remove(path)


def test_read_wav_as_float_stereo_averages_channels():
  '''Stereo input is downmixed to mono by averaging the two channels.'''
  n = 100
  left = np.full(n, 10000, dtype=np.int16)
  right = np.full(n, 20000, dtype=np.int16)
  stereo = np.column_stack((left, right))
  path = _write_temp_wav(stereo)
  try:
    result, _, _ = common.read_wav_as_float(path)
    assert result.ndim == 1
    assert len(result) == n
    # Mean of 10000 and 20000 is 15000 → 15000 / 32767.
    assert result[0] == pytest.approx(15000.0 / 32767.0, rel=1e-3)
  finally:
    os.remove(path)


def test_read_wav_as_float_float32_passthrough():
  '''float32 WAV values are returned unchanged as float32.'''
  data = np.array([0.5, -0.5, 1.0, -1.0], dtype=np.float32)
  path = _write_temp_wav(data)
  try:
    result, _, orig_dtype = common.read_wav_as_float(path)
    assert orig_dtype == np.dtype('float32')
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, data, rtol=1e-5)
  finally:
    os.remove(path)


def test_read_wav_as_float_file_not_found():
  '''Raises FileNotFoundError for a nonexistent path.'''
  with pytest.raises(FileNotFoundError):
    common.read_wav_as_float('/nonexistent/path/audio.wav')


# --- get_active_signal_rms ---

def test_get_active_signal_rms_silent():
  '''All-zero signal returns 0.0.'''
  data = np.zeros(1000, dtype=np.float32)
  assert common.get_active_signal_rms(data) == 0.0


def test_get_active_signal_rms_constant():
  '''Constant-amplitude signal returns that amplitude as RMS.'''
  data = np.full(1000, 0.5, dtype=np.float32)
  assert common.get_active_signal_rms(data) == pytest.approx(0.5, rel=1e-5)


def test_get_active_signal_rms_sine():
  '''Sine wave RMS is close to amplitude / sqrt(2).'''
  t = np.linspace(0, 1, 44100, endpoint=False)
  amplitude = 0.8
  data = amplitude * np.sin(2 * np.pi * 440 * t)
  result = common.get_active_signal_rms(data)
  assert result == pytest.approx(amplitude / np.sqrt(2), rel=1e-2)


def test_get_active_signal_rms_trims_silence():
  '''Leading and trailing silence is excluded from the RMS calculation.'''
  silence = np.zeros(1000, dtype=np.float64)
  active = np.ones(100, dtype=np.float64)
  signal = np.concatenate([silence, active, silence])
  # With the default -40 dB threshold, only the active block (value 1.0)
  # is above threshold; its RMS must be 1.0.
  assert common.get_active_signal_rms(signal) == pytest.approx(1.0, rel=1e-5)


# --- prepend_silence ---

def test_prepend_silence_mono_length():
  '''Prepended silence adds the correct number of samples for mono data.'''
  fs = 44100
  duration_s = 0.1
  n_silence = int(fs * duration_s)
  data = np.ones(200, dtype=np.float32)
  result = common.prepend_silence(data, fs, duration_s)
  assert len(result) == 200 + n_silence


def test_prepend_silence_mono_leading_zeros():
  '''The prepended region is all zeros for mono data.'''
  fs = 44100
  duration_s = 0.05
  n_silence = int(fs * duration_s)
  data = np.ones(100, dtype=np.float32)
  result = common.prepend_silence(data, fs, duration_s)
  np.testing.assert_array_equal(result[:n_silence], 0)


def test_prepend_silence_stereo_shape():
  '''Prepended silence has the correct 2-D shape for stereo data.'''
  fs = 44100
  duration_s = 0.1
  n_silence = int(fs * duration_s)
  data = np.ones((200, 2), dtype=np.int16)
  result = common.prepend_silence(data, fs, duration_s)
  assert result.shape == (200 + n_silence, 2)
  np.testing.assert_array_equal(result[:n_silence, :], 0)


def test_prepend_silence_preserves_dtype():
  '''The output dtype matches the input dtype.'''
  data = np.array([1, 2, 3], dtype=np.int16)
  result = common.prepend_silence(data, sample_rate=44100, duration_s=0.01)
  assert result.dtype == np.int16


# --- get_scaled_vcv_data ---

def test_get_scaled_vcv_data_non_wav_raises():
  '''A non-.wav path raises ValueError before any file I/O.'''
  with pytest.raises(ValueError, match=r'\.wav'):
    common.get_scaled_vcv_data('file.mp3', target_db_spl=60.0, ref_db_spl=65.0)


def test_get_scaled_vcv_data_gain_applied():
  '''Output amplitude reflects the expected linear gain factor.'''
  fs = 44100
  # Use a mid-range amplitude so 6 dB gain does not clip.
  n = int(0.1 * fs)
  data = np.full(n, 8000, dtype=np.int16)
  path = _write_temp_wav(data, fs)
  try:
    delta_db = 6.0
    scaled, out_fs = common.get_scaled_vcv_data(
        path, target_db_spl=65.0 + delta_db, ref_db_spl=65.0
    )
    assert out_fs == fs
    gain = 10 ** (delta_db / 20.0)
    output_float = scaled.astype(np.float32) / 32767.0
    input_float = data.astype(np.float32) / 32767.0
    np.testing.assert_allclose(
        output_float,
        np.clip(input_float * gain, -1.0, 1.0),
        rtol=1e-3,
    )
  finally:
    os.remove(path)


def test_get_scaled_vcv_data_file_not_found():
  '''FileNotFoundError is re-raised for a missing WAV file.'''
  with patch('common.st'):
    with pytest.raises(FileNotFoundError):
      common.get_scaled_vcv_data(
          '/nonexistent_file.wav', target_db_spl=65.0, ref_db_spl=65.0
      )
