"""Minimal unit tests for audio_playback conversion function."""

import unittest
import os
import tempfile
import numpy as np
import scipy.io.wavfile as wavfile

# Assuming audio_playback.py is accessible
import audio_converter


# Helper function to create a dummy stereo WAV file
def create_dummy_stereo_wav(filename, sr=44100, duration=0.1, dtype=np.int16):
  """Creates a simple stereo WAV file for testing."""
  samples = int(sr * duration)
  t = np.arange(samples) / sr
  # Create distinct left/right channels
  data_left = (np.sin(2 * np.pi * 440 * t) * 32760).astype(dtype)
  data_right = (np.sin(2 * np.pi * 880 * t) * 32760).astype(dtype)
  stereo_data = np.column_stack((data_left, data_right))
  wavfile.write(filename, sr, stereo_data)
  return sr, stereo_data


# Helper function to create a dummy mono WAV file
def create_dummy_mono_wav(filename, sr=44100, duration=0.1, dtype=np.int16):
  """Creates a simple mono WAV file for testing."""
  samples = int(sr * duration)
  mono_data = (np.sin(2 * np.pi * 440 * np.arange(samples) / sr) * 32760)\
    .astype(dtype)
  wavfile.write(filename, sr, mono_data)
  return sr, mono_data


class TestAudioConversion(unittest.TestCase):
  """Test cases for audio_playback.convert_stereo_to_mono_wav."""

  def setUp(self):
    """Create temporary files for testing."""
    # Create a dummy stereo WAV file
    handle, path = tempfile.mkstemp(suffix='.wav')
    os.close(handle)
    self.stereo_filepath = path
    self.sr_stereo, self.stereo_data = create_dummy_stereo_wav(path)

    # Create a dummy mono WAV file
    handle, path = tempfile.mkstemp(suffix='.wav')
    os.close(handle)
    self.mono_filepath = path
    self.sr_mono, self.mono_data = create_dummy_mono_wav(path)

    # Dummy path for non-existent file test
    self.non_existent_filepath = os.path.join(
        tempfile.gettempdir(), 'non_existent_test_file.wav'
    )
    if os.path.exists(self.non_existent_filepath):
      os.remove(self.non_existent_filepath)

  def tearDown(self):
    """Clean up temporary files."""
    if os.path.exists(self.stereo_filepath):
      os.remove(self.stereo_filepath)
    if os.path.exists(self.mono_filepath):
      os.remove(self.mono_filepath)

  def test_convert_keep_left(self):
    """Test converting stereo to keep only the left channel."""
    fs, modified_data = audio_converter.convert_stereo_to_mono_wav(
        self.stereo_filepath, 'left'
    )
    self.assertEqual(fs, self.sr_stereo)
    self.assertEqual(modified_data.shape, self.stereo_data.shape)
    # Check right channel (index 1) is zero
    self.assertTrue(np.all(modified_data[:, 1] == 0))
    # Check left channel (index 0) is unchanged
    np.testing.assert_array_equal(modified_data[:, 0], self.stereo_data[:, 0])

  def test_convert_keep_right(self):
    """Test converting stereo to keep only the right channel."""
    fs, modified_data = audio_converter.convert_stereo_to_mono_wav(
        self.stereo_filepath, 'right'
    )
    self.assertEqual(fs, self.sr_stereo)
    self.assertEqual(modified_data.shape, self.stereo_data.shape)
    # Check left channel (index 0) is zero
    self.assertTrue(np.all(modified_data[:, 0] == 0))
    # Check right channel (index 1) is unchanged
    np.testing.assert_array_equal(modified_data[:, 1], self.stereo_data[:, 1])

  def test_convert_mono_input_keep_left(self):
    """Test converting mono input (keeping left)."""
    fs, modified_data = audio_converter.convert_stereo_to_mono_wav(
        self.mono_filepath, 'left'
    )
    self.assertEqual(fs, self.sr_mono)
    # Check shape is now stereo
    self.assertEqual(modified_data.ndim, 2)
    self.assertEqual(modified_data.shape[1], 2)
    # Check right channel (index 1) is zero
    self.assertTrue(np.all(modified_data[:, 1] == 0))
    # Check left channel (index 0) equals original mono data
    np.testing.assert_array_equal(modified_data[:, 0], self.mono_data)

  def test_invalid_ear(self):
    """Test error handling for invalid ear argument."""
    with self.assertRaisesRegex(ValueError,
                                "ear_to_keep must be 'left' or 'right'."):
      audio_converter.convert_stereo_to_mono_wav(self.stereo_filepath, 'both')

  def test_file_not_found(self):
    """Test error handling for non-existent file."""
    with self.assertRaises(FileNotFoundError):
      audio_converter.convert_stereo_to_mono_wav(
          self.non_existent_filepath, 'left'
      )


if __name__ == '__main__':
  unittest.main()
