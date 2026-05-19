"""Generic signal processing tools relevant to the human perception of audio."""

import numpy as np
import scipy.signal as signal

def dbspl_to_amplitude(dbspl: float | np.ndarray,
                       fs_db: float = 94.0) -> float | np.ndarray:
  """Converts dB SPL (sound pressure level) to RMS amplitude.

  Args:
    dbspl: The dB SPL value(s) to convert to amplitude.
    fs_db: The full-scale dB level, which is the dB SPL corresponding to
           an RMS amplitude of 1. Defaults to 94 dB SPL, because an RMS
           amplitude of 1 Pascal corresponds to 94 dB SPL.

  Returns:
    The RMS amplitude value(s).
  """
  return 10 ** ((dbspl - fs_db) / 20)

def amplitude_to_dbspl(amplitude: float, fs_db: float = 94.0) -> float:
  """Converts RMS amplitude to dB SPL (sound pressure level).

  Args:
    amplitude: The RMS amplitude of the signal
    fs_db: The full-scale dB level, which is the dB SPL corresponding to
           an RMS amplitude of 1. Defaults to 94 dB SPL, because an RMS
           amplitude of 1 Pascal corresponds to 94 dB SPL.

  Returns:
    The dB SPL value.
  """
  return 20 * np.log10(amplitude) + fs_db

def cus_to_sones(cus: float, cu_factor: float = 13.0,
                 cu_exponent: float = 0.3) -> float:
  """Converts categorical units of loudness (CUs) to sones.

  A sone of 1 is defined as the loudness of a 1 kHz tone at 40 dB SPL above the
  listener's threshold.

  Args:
      cus: The loudness value in CUs.
      cu_factor: The CU factor.
      cu_exponent: The CU exponent.

  Returns:
      The loudness value in sones.
  """
  return (cus / cu_factor) ** (1 / cu_exponent)

def sones_to_cus(sones: float, cu_factor: float = 13.0,
                 cu_exponent: float = 0.3) -> float:
  """Converts sones to categorical units of loudness (CUs).

  Args:
      sones: The loudness value in sones.
      cu_factor: The CU factor.
      cu_exponent: The CU exponent.

  Returns:
      The loudness value in CUs.
  """
  return cu_factor * sones ** cu_exponent

def phons_to_sones(phons: float) -> float:
  """Converts phons to sones."""
  return 2 ** ((phons - 40) / 10)

def sones_to_phons(sones: float) -> float:
  """Converts sones to phons."""
  return 10 * np.log(sones) / np.log(2) + 40

def cf_to_audf(cf_hz: np.ndarray, order: int = 9
               ) -> tuple[np.ndarray, np.ndarray]:
  """Converts frequencies to an auditory-like frequency scale.

  This function computes an auditory-like frequency scale offset to 0 at 1 kHz,
  ranging within -1 to +1.7 to cover 20 Hz to 20 kHz. It can also optionally
  return powers of this scale for use in polynomial models.

  Args:
      cf_hz: Center frequencies in Hz.
      order: The highest order of the audf powers to be calculated.

  Returns:
      A tuple containing:
          - The auditory-like frequency scale
          - A matrix of powers of `audf`
  """
  # Start with CAM scale parameters from
  # Chen, Zhangli, Guangshu Hu, Brian R. Glasberg, and Brian CJ Moore.
  # "A new method of calculating auditory excitation patterns and loudness
  # for steady sounds." Hearing research 282, no. 1-2 (2011): 204-215.
  break_f = 228.8  # 1/0.00437.
  high_q = 9.294  # 21.4/log(10).
  cams = high_q * np.log(cf_hz / break_f + 1)
  cam1000 = high_q * np.log(1000 / break_f + 1)
  # Normalize to a reasonable signed range for polynomial fitting.
  audf = cams / cam1000 - 1

  audf_powers = np.zeros((len(audf), order + 1))
  audf_powers[:, 0] = 1  # First column is 0 power; avoid 0^0
  for exponent in range(1, order + 1):
    audf_powers[:, exponent] = audf ** exponent

  return audf, audf_powers

def tukey_window(tone, alpha=0.1):
  """Applies a Tukey window to the signal.

  Args:
    tone: The tone to window.
    alpha: The Tukey window parameter. When alpha=0, the Tukey window is a
           rectangular window (with no effect), and when alpha=1, it is
           equivalent to a Hann window.

  Returns:
    The windowed tone.
  """
  return tone * signal.windows.tukey(len(tone), alpha)
