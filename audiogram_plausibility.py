"""Checks whether a manually entered audiogram has a plausible shape.

Proof of concept. Fits the global parametric hearing-loss model used by the
adaptive PTA (`hearing_models.hearing_level_model`) to an audiogram and reports
how well it fits. A typed-in audiogram the model cannot follow is suspect.

The fit is a linear least-squares fit of the model's four basis shapes, with
the first (overall loss) coefficient constrained to be non-negative as the
model specifies. The PTA's global dictionary search
(`logistic_fit.GlobalDictionaryFitter`) uses much narrower ranges (c3 and c4
within +/-0.5); on the NAL clinical audiograms those ranges put most fits on a
bound and raise the median RMS error from 3.4 to 5.5 dB, so they are not used.

The fit only measures how far the entered points are from the model. With few
frequencies the fitted curve can swing between them, so it is not meant for
interpolating or extrapolating thresholds.
"""

import dataclasses

import numpy as np
from scipy import optimize

from hearing_models import hearing_level_model

COEFF_LOWER = np.array([0.0, -np.inf, -np.inf, -np.inf])
COEFF_UPPER = np.full(4, np.inf)

# Accepted range of an entered frequency (Hz).
MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ = 125.0, 8000.0
# Accepted range of an entered threshold (dB HL). The model has no shapes
# better than 0 dB HL, so thresholds below 0 count fully as fit error: a flat
# -10 dB HL audiogram at 8 frequencies is rejected (RMS 9.9 dB). No NHANES ear
# had every threshold at -5 dB HL or better.
MIN_DBHL, MAX_DBHL = -10.0, 120.0
# RMS fit-error limit, chosen to reject under 2% of real audiograms. It rejects
# 1.5% of 8,566 NHANES ears (2015-16, adults 20-69, 7 frequencies from 500 Hz
# to 8 kHz) and none of the 106 NAL clinical ears (8 frequencies). The NHANES
# rate leaves margin because two ears per person shrink the effective sample,
# and the false-alarm rate with 8 frequencies is likely a little higher. What
# it misses: in a synthetic test, about a quarter of single-frequency errors of
# 40-60 dB (e.g. 55 typed as 5); real precipitous losses and such errors look
# alike to the model. With fewer than 8 frequencies the fit absorbs more of
# the noise, so the limit is a little more lenient (about 11.5 dB effective at
# 6 frequencies).
MAX_RMS_ERROR_DB = 9.5
# The four-frequency-average frequencies, present on almost every audiogram.
REQUIRED_FREQUENCIES_HZ = (500, 1000, 2000, 4000)
# The model has 4 coefficients; with few more points it can fit anything. So
# at least two frequencies besides the required ones are needed.
MIN_FREQUENCIES = 6


@dataclasses.dataclass
class GlobalFit:
  """Result of fitting the global model to one audiogram."""
  freqs_hz: np.ndarray
  measured_dbhl: np.ndarray
  fitted_dbhl: np.ndarray
  coeffs: np.ndarray

  @property
  def residuals_db(self) -> np.ndarray:
    return self.measured_dbhl - self.fitted_dbhl

  @property
  def rms_error_db(self) -> float:
    return float(np.sqrt(np.mean(self.residuals_db**2)))

  @property
  def max_abs_error_db(self) -> float:
    return float(np.max(np.abs(self.residuals_db)))


def basis_shapes(freqs_hz: np.ndarray) -> np.ndarray:
  """The model's four basis shapes at freqs_hz, one per column (dB HL)."""
  return np.column_stack([
      hearing_level_model(freqs_hz, np.eye(4)[k])[0].ravel()
      for k in range(4)
  ])


def fit_global_model(audiogram: dict[float, float]) -> GlobalFit:
  """Fits the global model to an audiogram {frequency Hz: dB HL}."""
  keys = sorted(audiogram)
  freqs = np.array(keys, dtype=float)
  dbhl = np.array([audiogram[k] for k in keys], dtype=float)
  basis = basis_shapes(freqs)
  coeffs = optimize.lsq_linear(basis, dbhl,
                               bounds=(COEFF_LOWER, COEFF_UPPER)).x
  return GlobalFit(freqs, dbhl, basis @ coeffs, coeffs)


def is_plausible(audiogram: dict[float, float]) -> bool:
  """True if an entered audiogram {frequency Hz: dB HL} has a plausible shape.

  Raises:
    ValueError: if a frequency is outside MIN_FREQUENCY_HZ to
      MAX_FREQUENCY_HZ, any of REQUIRED_FREQUENCIES_HZ is missing, or fewer
      than MIN_FREQUENCIES frequencies are given.
  """
  # Written so that a NaN frequency also fails.
  bad = [f for f in audiogram
         if not MIN_FREQUENCY_HZ <= f <= MAX_FREQUENCY_HZ]
  if bad:
    raise ValueError(f"Frequencies must be {MIN_FREQUENCY_HZ:g} to "
                     f"{MAX_FREQUENCY_HZ:g} Hz, got {bad}.")
  missing = [f for f in REQUIRED_FREQUENCIES_HZ if f not in audiogram]
  if missing:
    raise ValueError(f"Missing required frequencies (Hz): {missing}.")
  if len(audiogram) < MIN_FREQUENCIES:
    raise ValueError(f"Need at least {MIN_FREQUENCIES} frequencies, "
                     f"got {len(audiogram)}.")
  values = np.array(list(audiogram.values()), dtype=float)
  if np.any(~np.isfinite(values) | (values < MIN_DBHL) | (values > MAX_DBHL)):
    return False
  return fit_global_model(audiogram).rms_error_db <= MAX_RMS_ERROR_DB
