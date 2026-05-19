"""A module for synthesizing and manipulating audio for psychoacoustic tests."""

import numpy as np
from scipy.io import wavfile
import warnings
from scipy import signal

import common


def _calculate_rms(audio_data: np.ndarray) -> float:
  """Calculates the Root-Mean-Square of a NumPy array."""
  return np.sqrt(np.mean(audio_data**2))


def basic_speech_shaped_noise(
    white_noise: np.ndarray, sample_rate: int
) -> np.ndarray:
  """Filter white noise to generate speech-shaped noise.

  This function uses a simple Butterworth filter to model the long-term average
  speech spectrum.

  Args:
    white_noise: A NumPy array containing the white noise to be shaped.
    sample_rate: The sample rate of the noise in Hz.

  Returns:
    A NumPy array containing the speech-shaped noise.
  """
  # Use a low-pass filter to approximate the speech spectrum roll-off.
  # A 4th order Butterworth filter with a cutoff at 1.5 kHz is a reasonable
  # and simple approximation.
  b, a = signal.butter(4, 1500, btype='low', fs=sample_rate)
  shaped_noise = signal.lfilter(b, a, white_noise)
  return shaped_noise


def advanced_speech_shaped_noise(white_noise: np.ndarray, sample_rate: int):
  """Filter white noise to generate speech-shaped noise.

  This function uses a pole-zero approximation to model the long-term average
  speech spectrum. The method is computationally efficient and creates
  perceptually realistic speech-shaped noise.

  Args:
      white_noise (np.ndarray): The white noise signal to be shaped.
      sample_rate (int): The sampling rate in Hz (e.g., 44100). Must be at
        least 32000 Hz to ensure an accurate filter response.

  Returns:
      numpy.ndarray: A 1D array of speech-colored noise, normalized to [-1, 1].
  """
  if sample_rate < 32000:
    raise ValueError(
        'Sample rate must be at least 32000 Hz to ensure an accurate '
        'filter frequency response.'
    )
  # Design the main speech-shaping IIR filter, using a pole-zero model in the
  # continuous-time s-plane, which is then converted to the discrete-time
  # z-plane for digital filtering.
  pole_freq_hz = 600
  # A damping factor zeta < 1/sqrt(2) creates a resonant peak.
  zeta = 0.65
  # Calculate the s-plane pole location.
  s_pole = 2 * np.pi * pole_freq_hz * (-zeta + 1j * np.sqrt(1 - zeta ** 2))
  # A single real zero is used to slow the roll-off to -6 dB/octave.
  zero_freq_hz = 1200
  s_zero = -2 * np.pi * zero_freq_hz

  # Convert from s-plane to z-plane.
  # We map the continuous-time poles/zeros to their discrete-time equivalents
  # using the formula z = exp(s*T), where T is the sampling period.
  z_pole = np.exp(s_pole / sample_rate)
  z_zero = np.exp(s_zero / sample_rate)

  # Create the digital filter coefficients.
  # The denominator 'a' coefficients are derived from the poles.
  a_coeffs = np.array([1, -2 * np.real(z_pole), np.abs(z_pole) ** 2])
  # The numerator 'b' coefficients are derived from the zeros.
  b_coeffs = np.array([1, -z_zero])
  # Normalize the filter to have unity (0 dB) gain at DC (0 Hz). This makes
  # the overall signal level more predictable. Gain at DC corresponds to the
  # transfer function evaluated at z=1.
  b_coeffs *= np.sum(a_coeffs) / np.sum(b_coeffs)

  # Apply the shaping filter to the input white noise.
  shaped_noise = signal.lfilter(b_coeffs, a_coeffs, white_noise)

  # Design and apply an AC-coupling filter. This is a simple high-pass filter
  # to remove any DC offset from the signal.
  ac_freq_hz = 100  # Corner frequency of the high-pass filter.
  s_ac = -2 * np.pi * ac_freq_hz
  z_ac = np.exp(s_ac / sample_rate)
  # This filter has a zero right at DC (z=1) and a pole at the corner frequency.
  b_ac = np.array([1, -1])
  a_ac = np.array([1, -z_ac])
  # Apply the AC-coupling filter to the already-shaped noise.
  final_noise = signal.lfilter(b_ac, a_ac, shaped_noise)

  # Normalize the final output to have a peak amplitude of 1.0.
  final_noise /= np.max(np.abs(final_noise))
  return final_noise


def mix_vcv_with_noise(
    snr_db: float,
    ear: str = 'both',
    noise_type: str = 'White Noise',
    audio_data: np.ndarray = None,
    sample_rate: int = None,
    clean_vcv_path: str = None,
    target_output_rms: float = None,
    ramp_duration_s: float = 0.0
) -> tuple[np.ndarray, int]:
  """Mixes a clean VCV audio file with noise at a specified SNR.

  The level of the original VCV signal is preserved unless target_output_rms
  is specified. The noise is scaled to achieve the target SNR relative to the
  signal. If the combined signal exceeds the maximum possible amplitude, it is
  clipped and a warning is issued (unless normalization brings it into range).

  Args:
    snr_db: The desired Signal-to-Noise Ratio in decibels.
    ear: The ear to present the stimulus to ('left', 'right', or 'both').
      If 'left' or 'right', the opposite channel will be silenced.
    noise_type: The type of noise to use ('White Noise',
      'Basic Speech-Shaped Noise' or 'Advanced Speech-Shaped Noise').
    audio_data: Optional pre-loaded audio data as a NumPy array.
    sample_rate: Optional sample rate if audio_data is provided.
    clean_vcv_path: The file path to the clean (noise-free) VCV .wav file.
      This is optional if audio_data and sample_rate are provided.
    target_output_rms: Optional target RMS amplitude for the final mixed signal.
      If provided, the entire mix (Signal + Noise) is scaled so its RMS matches
      this value. This prevents clipping at low SNRs and ensures constant
      loudness.
    ramp_duration_s: Optional duration in seconds for a half-cosine ramp to be
      applied to the start of the noise.

  Returns:
    A tuple containing:
    - A NumPy array of the mixed audio data, in the original datatype.
    - The sample rate of the audio file.
  """
  # Read the clean audio file if not provided directly.
  if audio_data is None or sample_rate is None:
    if clean_vcv_path is None:
      raise ValueError('Must provide either audio_data or clean_vcv_path.')
    sample_rate, audio_data = wavfile.read(clean_vcv_path)

  original_dtype = audio_data.dtype

  # Convert integer audio to a normalized float format between [-1.0, 1.0].
  # This allows for consistent signal processing. Floats are passed through.
  if np.issubdtype(original_dtype, np.integer):
    max_val = np.iinfo(original_dtype).max
    signal_float = audio_data.astype(np.float64) / max_val
  else:
    signal_float = audio_data.astype(np.float64)

  # Calculate signal RMS. This signal is now guaranteed to be mono.
  signal_rms = common.get_active_signal_rms(signal_float)
  if signal_rms == 0:
    warnings.warn('Input signal is silent, cannot apply SNR.')
    return audio_data, sample_rate  # Return original silent audio.

  # Generate a single source of white noise.
  white_noise = np.random.randn(len(signal_float))

  # Shape the noise based on the selected type.
  if noise_type == 'Basic Speech-Shaped Noise':
    noise = basic_speech_shaped_noise(white_noise, sample_rate)
  elif noise_type == 'Advanced Speech-Shaped Noise':
    noise = advanced_speech_shaped_noise(white_noise, sample_rate)
  else:  # Default to White Noise
    noise = white_noise

  # Calculate required noise RMS and scale the noise.
  # First, normalize the generated noise to have an RMS of 1.0
  noise_rms = _calculate_rms(noise)
  if noise_rms > 0:
    noise_normalized = noise / noise_rms
  else:
    noise_normalized = noise # Avoid division by zero for silent noise

  noise_rms_target = signal_rms / (10 ** (snr_db / 20))
  noise_scaled = noise_normalized * noise_rms_target

  # Apply a ramp to the noise if requested.
  if ramp_duration_s > 0:
    ramp_samples = int(ramp_duration_s * sample_rate)
    # Ensure ramp isn't longer than the signal itself.
    ramp_samples = min(ramp_samples, len(noise_scaled))
    if ramp_samples > 0:
      # Create a half-cosine ramp (0 to 1).
      t = np.linspace(0, np.pi, ramp_samples)
      ramp_on = (1 - np.cos(t)) / 2
      ramp_off = (1 + np.cos(t)) / 2
      noise_scaled[:ramp_samples] *= ramp_on
      noise_scaled[-ramp_samples:] *= ramp_off

  # Mix signal and noise.
  mixed_signal_float = signal_float + noise_scaled

  # Apply output normalization if requested.
  if target_output_rms is not None:
    # Calculate current RMS of the mix.
    current_mix_rms = _calculate_rms(mixed_signal_float)
    if current_mix_rms > 0:
      normalization_gain = target_output_rms / current_mix_rms
      mixed_signal_float *= normalization_gain

  # Spatialize the mixed mono audio to a stereo signal first.
  stereo_output_float = np.repeat(mixed_signal_float[:, np.newaxis], 2, axis=1)
  if ear == 'left':
    stereo_output_float[:, 1] = 0  # Zero out right channel.
  elif ear == 'right':
    stereo_output_float[:, 0] = 0  # Zero out left channel.

  # Handle potential clipping.
  if np.max(np.abs(stereo_output_float)) > 1.0:
    warnings.warn(
        f'Clipping occurred for SNR = {snr_db} dB. '
        f'Max absolute value was {np.max(np.abs(stereo_output_float)):.2f}.'
    )
    stereo_output_float = np.clip(stereo_output_float, -1.0, 1.0)

  # Convert back to original datatype.
  if np.issubdtype(original_dtype, np.integer):
    mixed_audio_data = (stereo_output_float * max_val).astype(original_dtype)
  else:
    mixed_audio_data = stereo_output_float.astype(original_dtype)

  return mixed_audio_data, sample_rate
