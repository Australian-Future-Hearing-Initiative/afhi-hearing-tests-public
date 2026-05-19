"""Tests for the audio_tools module."""

import numpy as np
import unittest

import audio_tools

class TestAudioTools(unittest.TestCase):
  """Tests for all functions in audio_tools."""

  def test_amplitude_to_dbspl(self):
    """Tests the amplitude_to_dbspl function."""
    # 1 Pascal should be 94 dB SPL, by definition.
    self.assertAlmostEqual(audio_tools.amplitude_to_dbspl(1.0), 94.0)
    self.assertAlmostEqual(
        audio_tools.amplitude_to_dbspl(0.1), 74.0
    )  # 0.1 Pascal should be 74 dB SPL.
    self.assertAlmostEqual(
        audio_tools.amplitude_to_dbspl(1.0, fs_db=100.0), 100.0
    )  # Test with different fs_db.

  def cus_to_sones_conversion(self):
    """Tests back and forward conversion between CUs and sones."""
    # 1 sone should be 1.0 CUs.
    self.assertAlmostEqual(audio_tools.cus_to_sones(1.0), 1.0)
    self.assertAlmostEqual(audio_tools.sones_to_cus(1.0), 1.0)
    # Test for an arbitrary number that converting forward and back
    # preserves the value.
    sones = 42.0
    cus = audio_tools.sones_to_cus(sones)
    self.assertAlmostEqual(audio_tools.cus_to_sones(cus), sones)

  def sones_to_phons_conversion(self):
    """Tests back and forward conversion between sones and phons."""
    # 1 sone should be 40 phons.
    self.assertAlmostEqual(audio_tools.sones_to_phons(1.0), 40.0)
    self.assertAlmostEqual(audio_tools.phons_to_sones(40.0), 1.0)
    # Test for an arbitrary number that converting forward and back
    # preserves the value.
    phons = 60.0
    sones = audio_tools.phons_to_sones(phons)
    self.assertAlmostEqual(audio_tools.sones_to_phons(sones), phons)

  def test_cf_to_audf(self):
    """Tests the cf_to_audf function."""
    # For an input of 1000 Hz, output should be 0.
    audf, audf_powers = audio_tools.cf_to_audf(np.array([1000]))
    self.assertAlmostEqual(audf, 0)
    # Check that the powers of audf are correct for trivial input.
    self.assertTrue((audf_powers == np.array(
      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])).all())
    # Check that the powers make sense for non-trivial input.
    audf, audf_powers = audio_tools.cf_to_audf(np.array([2000]))
    self.assertAlmostEqual(audf_powers[0, 1], audf[0])
    self.assertAlmostEqual(audf_powers[0, 2], audf**2)
    self.assertAlmostEqual(audf_powers[0, 9], audf**9)

if __name__ == '__main__':
  unittest.main()
