"""A 1D Bayesian adaptive estimator using the ZEST procedure."""

import math
import numpy as np
from scipy.stats import norm

# --- Configuration for the VCV 10-AFC Test ---
# CHANCE_RATE (gamma): The probability of a correct answer when the signal is
# imperceptible. Since this is a 10-Alternative Forced Choice (10-AFC) task,
# the chance rate is 0.10
CHANCE_RATE = 0.10
# LAPSE_RATE (lambda): The probability of an incorrect answer when the signal
# is very clear (e.g., due to a misclick or lapse in attention).
LAPSE_RATE = 0.02
# ASSUMED_SLOPE (beta): The assumed slope of the psychometric function.
ASSUMED_SLOPE = 1.0

# Define the discrete grid for the threshold (alpha).
MIN_SNR_DB = -55.0
MAX_SNR_DB = 40.0
STEP_SNR_DB = 0.5
THRESHOLD_GRID = np.arange(MIN_SNR_DB, MAX_SNR_DB + STEP_SNR_DB, STEP_SNR_DB)

# Default Prior settings.
PRIOR_SD = 20.0

CONSONANT_LABELS = {
    'B': 'aba', 'D': 'ada', 'G': 'aga', 'K': 'aka',
    'N': 'ana', 'S': 'asa', 'SH': 'asha', 'T': 'ata',
    'V': 'ava', 'Z': 'aza',
    # Extended consonants for the custom upload feature.
    'M': 'ama', 'P': 'apa', 'F': 'afa',
    'TH': 'atha', 'DH': 'adha', 'ZH': 'azha'
}

# Consonant class definitions. Each class specifies:
#   - members: the consonants in this class.
#   - floor_db: minimum presented SNR. Prevents the adaptive
#     algorithm from going so low that the consonant is
#     inaudible (causing blind guessing).
#   - initial_snr_db: the prior mean for the ZEST estimator.
#     Set deliberately above the expected threshold so
#     participants start with audible stimuli and the
#     algorithm naturally descends to threshold.
# These values are derived from empirical test data.
CONSONANT_CLASSES = {
    'C1': {
        'members': ['B', 'V', 'M', 'TH', 'DH', 'F'],
        'floor_db': -6.0,
        'initial_snr_db': 15.0,
    },
    'C2': {
        'members': ['Z', 'T', 'S', 'SH', 'ZH'],
        'floor_db': -18.0,
        'initial_snr_db': 5.0,
    },
    'C3': {
        'members': ['N', 'D', 'K', 'G', 'P'],
        'floor_db': -12.0,
        'initial_snr_db': 10.0,
    },
}

# Display order for consonant classes in plots and tables.
CLASS_DISPLAY_ORDER = ['C2', 'C3', 'C1']

# Flat consonant order derived from CLASS_DISPLAY_ORDER.
ORDERED_CONSONANTS = [
    c
    for cls in CLASS_DISPLAY_ORDER
    for c in CONSONANT_CLASSES[cls]['members']
]

# Build per-consonant lookups from the class definitions.
CONSONANT_SNR_FLOOR_DB = {}
CONSONANT_INITIAL_SNR_DB = {}
for _cls, _cfg in CONSONANT_CLASSES.items():
  for _c in _cfg['members']:
    CONSONANT_SNR_FLOOR_DB[_c] = _cfg['floor_db']
    CONSONANT_INITIAL_SNR_DB[_c] = _cfg['initial_snr_db']

# Sanity check: every consonant in CONSONANT_LABELS must
# belong to a class, and every class member must exist in
# CONSONANT_LABELS.
_class_consonants = {
    c for cfg in CONSONANT_CLASSES.values()
    for c in cfg['members']
}
_missing = (
    set(CONSONANT_LABELS.keys()) - _class_consonants
)
_extra = (
    _class_consonants - set(CONSONANT_LABELS.keys())
)
if _missing:
  raise ValueError(
      'Consonants missing from CONSONANT_CLASSES: '
      f'{sorted(_missing)}. Every consonant in '
      'CONSONANT_LABELS must belong to a class.'
  )
if _extra:
  raise ValueError(
      'Unknown consonants in CONSONANT_CLASSES: '
      f'{sorted(_extra)}. All class members '
      'must exist in CONSONANT_LABELS.'
  )


class ZestEstimator:
  """
  Implements the ZEST (Zippy Estimation by Sequential Testing) procedure.
  """
  def __init__(self, prior_mean=0.0, prior_sd=PRIOR_SD,
               slope=ASSUMED_SLOPE, chance_rate=CHANCE_RATE,
               lapse_rate=LAPSE_RATE):
    self.grid = THRESHOLD_GRID
    self.chance_rate = chance_rate
    self.lapse_rate = lapse_rate
    self.slope = slope
    self.history = []

    # Initialize the prior Probability Density Function (PDF) in log space.
    self.log_posterior = self._initialize_log_prior(prior_mean, prior_sd)

  def _initialize_log_prior(self, prior_mean, prior_sd):
    """Helper to initialize or reinitialize the log prior robustly."""
    log_prior = norm.logpdf(self.grid, loc=prior_mean, scale=prior_sd)
    return log_prior - np.logaddexp.reduce(log_prior)

  def reinitialize_prior(self, prior_mean, prior_sd):
    """Allows dynamic adjustment of the prior."""
    if not self.history:
      self.log_posterior = self._initialize_log_prior(prior_mean, prior_sd)

  @staticmethod
  def psychometric_function(snr, threshold, slope, chance_rate, lapse_rate):
    """Calculates P(Correct) using the logistic psychometric function."""
    exponent = -slope * (snr - threshold)
    p_raw = 1 / (1 + np.exp(np.clip(exponent, -700, 700)))
    p_correct = chance_rate + (1 - chance_rate - lapse_rate) * p_raw
    return p_correct

  def _get_posterior_pdf(self) -> np.ndarray:
    """Helper to safely convert log posterior back to a normalized PDF."""
    return np.exp(self.log_posterior)

  def get_next_snr(self) -> float:
    """
    Determines the optimal SNR for the next trial (mean of posterior).
    """
    posterior = self._get_posterior_pdf()
    mean_threshold = np.sum(posterior * self.grid)
    return mean_threshold

  def update(self, snr: float, is_correct: bool):
    """Updates the posterior distribution based on the trial outcome."""
    self.history.append((snr, is_correct))

    # 1. Calculate the Likelihood for every threshold value in the grid.
    p_correct_at_grid = self.psychometric_function(
        snr, self.grid, self.slope, self.chance_rate, self.lapse_rate
    )
    p_correct_at_grid = np.clip(p_correct_at_grid, 1e-20, 1.0 - 1e-20)

    log_likelihood = (np.log(p_correct_at_grid) if is_correct
                      else np.log(1.0 - p_correct_at_grid))

    # 2. Update the Posterior PDF in log space using Bayes' Theorem.
    self.log_posterior += log_likelihood

    # 3. Re-normalize the new Posterior PDF.
    self.log_posterior -= np.logaddexp.reduce(self.log_posterior)

  def get_estimate(self) -> tuple[float, float]:
    """
    Returns the current threshold estimate (mean) and uncertainty (SD).
    """
    posterior = self._get_posterior_pdf()
    mean_threshold = np.sum(posterior * self.grid)
    variance = np.sum(posterior * (self.grid - mean_threshold)**2)
    std_dev = math.sqrt(variance)
    return mean_threshold, std_dev
