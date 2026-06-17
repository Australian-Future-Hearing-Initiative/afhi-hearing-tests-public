"""Functions and constants associated with calibrating audio output."""

import numpy as np
import streamlit as st
import tempfile
import time
from scipy.io import wavfile

import common

CALIBRATION_TONE_FREQ = 1000  # Frequency of the calibration tone in Hz.
CALIBRATION_TONE_SPL = 70  # Calibration tone target volume for user.
CALIBRATION_TONE_DURATION = 20  # Duration of the calibration tone in seconds.
FS_HZ = 44100  # Sampling frequency in Hz used to synthesize the tone.
# Below is the amplitude that works out to be 70 dB SPL for Google Pixel Buds,
# tested in Jan 2025 using a sound level meter. The original value of 0.009
# requires the system volume to be set to maximum (tested on a MacBook Pro).
# This was updated to 0.23 to work with the volume set to 50% (tested based on
# a subjective comparison with the 0.009 amplitude) across all tones.
REFERENCE_AMPLITUDE = 0.23
REFERENCE_DB_SPL = 70

# Per-frequency offset (dB) to apply when using Airpods Pro 2 instead of
# Pixel Buds. Positive values mean the Airpods measure higher thresholds
# (i.e. the Airpods are quieter at that frequency relative to the Pixel Buds).
# IMPORTANT: All "Hearing Assistance" features (e.g. Conversation Boost,
# Loud Noise Reduction, Hearing Aid) must be disabled on the Airpods Pro 2
# for these offsets to be valid.
AIRPODS_PRO2_OFFSET = {
  250: -0.4,
  500: 0.5,
  1000: 3.0,
  2000: -0.1,
  3000: 1.0,
  4000: -0.3,
  6000: 1.7,
  8000: -0.5,
}
# Calibration values for VCVs.
REFERENCE_VCV_DB_SPL = 58  # This value was measured in the lab.
DEFAULT_VCV_DB_SPL = 65.0  # Default test level.

# Offset to make the synthetic VCVs match the loudness of the human ones.
SYNTHETIC_VCV_DB_SPL_OFFSET = 22.37
SYNTHETIC_VCV_LEADING_SILENCE_S = 0.25 # Leading silence for synthetic VCVs.


def play_calibration_tone(frequency_hz, volume, duration_s=0.5):
  """Generate and plays a tone to the user.

  Args:
    frequency_hz: The frequency of the tone in Hz.
    volume: The volume of the tone, between 0 and 1.
    duration_s: The duration of the tone in seconds.
  """
  # Ensure the volume is between 0 and 1.
  if volume < 0 or volume > 1:
    raise ValueError('Volume must be between 0 and 1.')
  # Generate the tone.
  t = np.linspace(0, duration_s, int(FS_HZ * duration_s))
  tone = np.sin(2 * np.pi * frequency_hz * t)
  tone = tone * volume  # Adjust the volume.
  # Convert to 16-bit integer format.
  tone = (tone * common.MAX_16_BIT_INT).astype(np.int16)
  # Save the tone to a temporary WAV file.
  with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
    wavfile.write(temp_file.name, FS_HZ, tone)
    common.autoplay_audio(temp_file.name)  # Play the temporary file.
  time.sleep(duration_s)  # Wait for audio to play.

def calibrate_audio():
  """Play a calibration tone at a standard frequency and amplitude.  """
  if not st.session_state.calibration_playing:
    amplitude = REFERENCE_AMPLITUDE
    st.session_state.calibration_playing = True
    play_calibration_tone(
      CALIBRATION_TONE_FREQ,
      amplitude,
      CALIBRATION_TONE_DURATION)
    st.session_state.calibration_playing = False
  else:
    st.session_state.calibration_playing = False

def dbhl_to_dbspl(
    db_hl: float, frequency_hz: int, configuration: str) -> float:
  """Converts dB HL to dB SPL using a reference table.

  The conversion is frequency-dependent, so we need to look up the value. It
  comes from both properties of the human hearing system, the testing
  algorithm used and the hardware.

  Args:
    db_hl: The hearing level in dB.
    frequency_hz: The frequency in Hz.
    configuration: The hw/sw configuration used in the experiment:
      'PixelBuds_HughsonWestlake', 'PixelBuds_Adaptive', or 'PixelBuds_Pip'.
  """
  if configuration == 'PixelBuds_HughsonWestlake':
    conversion_dict = {
      250: 14.3,
      500: 8.8,
      1000: 5.7,
      2000: 6.2,
      3000: 5.3,
      4000: 5.8,
      6000: 8.1,
      8000: 12.8,
    }
  elif configuration == 'PixelBuds_Adaptive':
    conversion_dict = {
      250: 8.9,
      500: 2.7,
      1000: -1.0,
      2000: 3.5,
      3000: -0.8,
      4000: 1.6,
      6000: 3.6,
      8000: 9.8,
    }
  elif configuration == 'PixelBuds_Pip':
    conversion_dict = {
      250: 5.6,
      500: 2.2,
      1000: 0.7,
      2000: 4.1,
      3000: -0.4,
      4000: 2.4,
      6000: 4.2,
      8000: 7.6,
    }
  else:
    raise ValueError('Hardware/test configuration not supported.')
  return db_hl + conversion_dict[frequency_hz]

def get_device_offset(frequency_hz: int, device: str) -> float:
  """Returns a calibration offset for the given device and frequency.

  This offset is applied on top of the existing PixelBuds-based calibration
  to adjust for a different headphone device.

  Args:
    frequency_hz: The frequency in Hz.
    device: The device name (from common.SUPPORTED_DEVICES).

  Returns:
    The offset in dB to subtract from the dB SPL value. A positive offset
    means the device is quieter at that frequency, so the output amplitude
    should be increased.
  """
  if device == common.DEVICE_PIXEL_BUDS:
    return 0.0
  elif device == common.DEVICE_AIRPODS_PRO2:
    return AIRPODS_PRO2_OFFSET.get(frequency_hz, 0.0)
  elif device == common.DEVICE_OTHER:
    # Use dynamically retrieved calibration offset from session state
    if 'dynamic_offsets' in st.session_state and st.session_state.dynamic_offsets:
      return st.session_state.dynamic_offsets.get(frequency_hz, 0.0)
    return 0.0
  else:
    raise ValueError(f'Unsupported device: {device}')


def dbspl_to_amp(db_spl: float,
                 ref_db_spl=REFERENCE_DB_SPL,
                 ref_amp=REFERENCE_AMPLITUDE) -> float:
  """Converts dB SPL to amplitude using a pair of reference values.

  Args:
    db_spl: The sound pressure level in dB.
    ref_db_spl: The reference sound pressure level in dB.
    ref_amp: The reference amplitude matching the reference SPL.

  Returns:
    The amplitude corresponding to the input dB SPL.
  """
  db_spl_relative = db_spl - ref_db_spl
  amplitude = ref_amp * 10 ** (db_spl_relative / 20)
  return amplitude

def amp_to_dbspl(amplitude: float, ref_db_spl=REFERENCE_DB_SPL,
                 ref_amp=REFERENCE_AMPLITUDE) -> float:
  """Converts amplitude to dB SPL using a pair of reference values.

  Args:
    amplitude: The amplitude of the signal.
    ref_db_spl: The reference sound pressure level in dB.
    ref_amp: The reference amplitude matching the reference SPL.

  Returns:
    The dB SPL value.
  """
  db_spl_relative = 20 * np.log10(amplitude / ref_amp)
  return ref_db_spl + db_spl_relative
