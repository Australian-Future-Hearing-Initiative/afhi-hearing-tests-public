"""Functions relating to the modelling of human hearing."""

import numpy as np

import audio_tools
import calibration

# Map button index to categorical units (CUs).
BUTTONS_TO_CUS = np.array([2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

# Minimum CU used when generating random test stimuli.
# CU 5 corresponds to "Very Soft", the lowest named
# button. This targets the estimated middle of the
# "Very Soft" category, hoping to see about 5% of
# responses in that extreme category and about 11% in
# each of the 8 categories between "Very Soft" and
# "Very Loud", and no "too loud". The extreme category
# responses don't need to be balanced.
MIN_STIMULUS_CU = 5

# Polynomial coefficients for the hearing level model. Each column represents
# a basis function for describing hearing loss shapes.
POLY_COEFFS = np.array([
    [39.2009, -10.1437, 4.3980, -2.1827],
    [27.1555, 20.2498, 11.6840, 4.1437],
    [14.6875, 33.1961, -21.5776, 21.0738],
    [-10.8151, -27.4179, -15.3929, -14.7968],
    [-3.8048, -33.5089, 16.4637, -23.0168],
    [3.3513, 24.6996, 0.4706, 13.7262]
])


def phons_to_dbspl(freqs_hz: np.ndarray, phon_levels: float | np.ndarray
) -> np.ndarray:
  """Estimates a constant loudness curve at specified frequencies.

  This function estimates a constant loudness curve at the specified
  frequencies and corresponding phon levels, using a polynomial function of
  frequency and a linear dB model.

  Args:
      freqs_hz: The n frequencies in Hz, as a column vector.
      phon_level: Phon levels, either as a scalar or an array of n values.

  Returns:
      An n-by-1 numpy array containing dB SPL values corresponding to the
      provided frequencies and phon levels.
  """
  # If phon_levels is a scalar make it an array of the same length as freqs_hz.
  if np.isscalar(phon_levels):
    phon_levels = np.full(freqs_hz.shape, phon_levels)
  elif len(phon_levels) != len(freqs_hz):
    raise ValueError('Phon levels must be scalar or match the number of freqs.')
  # Polynomial coefficients for the model.
  # TODO: Add some explanation regarding how to derive these coefficients.
  zero_and_rate_coeffs = np.array([
    [0.6725, 1.0063],
    [1.1465, 0.1293],
    [7.2111, -0.0947],
    [-123.9282, -0.0212],
    [25.3393, -0.1781],
    [246.3457, 0.5013],
    [-50.2157, -0.0901],
    [-190.0229, -0.4636],
    [88.6428, 0.2410]
  ])

  order = zero_and_rate_coeffs.shape[0] - 1
  _, audf_powers = audio_tools.cf_to_audf(freqs_hz.ravel(), order)
  zero_and_rate_approx = audf_powers @ zero_and_rate_coeffs
  zero_phon_curve = zero_and_rate_approx[:, 0]
  phon_rate_curve = zero_and_rate_approx[:, 1]
  # Make sure everything is a column vector.
  zero_phon_curve = zero_phon_curve.reshape(-1, 1)
  phon_rate_curve = phon_rate_curve.reshape(-1, 1)
  phon_levels = phon_levels.reshape(-1, 1)
  dbspl = zero_phon_curve + phon_levels * phon_rate_curve
  return dbspl

def dbhl_to_slopes(frequencies: np.ndarray, dbhl: np.ndarray) -> np.ndarray:
  """Estimates loudness growth slopes relative to normal hearing.

  This function use the parameterized model of hearing level (HL) to
  estimate loudness growth slopes relative to normal hearing at an arbitrary
  set of frequencies.

  Args:
      frequencies: The frequencies in Hz.
      dbhl: The dB HL values at the given frequencies.

  Returns:
      The loudness growth slopes relative to normal hearing.
  """
  db_threshold0 = phons_to_dbspl(frequencies, 0)  # dB SPL at 0 phon.
  db_threshold_hi = dbhl + db_threshold0  # For hearing-impaired.
  loud_sones = 24.0  # Nominal loud point, 86 phons, 24 sones.
  top_phons = audio_tools.sones_to_phons(loud_sones)
  db_loud = phons_to_dbspl(frequencies, top_phons)  # dB SPL at the loud point.
  inverse_slopes = (db_loud - db_threshold_hi) / (db_loud - db_threshold0)
  # Take care of unexpected extreme and negative cases.
  slopes = 1.0 / np.maximum(0.2, inverse_slopes)
  return slopes

def hearing_level_model(frequencies: np.ndarray, model_coeffs: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
  """Calculates dB HL (hearing level) at given frequencies.

  This function calculates dB HL at the given frequencies, with hearing loss
  parameterized by a row of 1 to 4 coefficients. The first coefficient is
  positive (0 for normal hearing), and subsequent coefficients are signed.

  Args:
      frequencies: The frequencies in Hz.
      model_coeffs: A numpy array of 1 to 4 coefficients representing the
                    hearing loss model.

  Returns:
      A tuple containing:
          - The dB HL values at the given frequencies.
          - The slopes of the hearing loss curves.
  """
  if len(model_coeffs) > 4 or len(model_coeffs) < 1:
    raise ValueError('Model coefficients must be 1 to 4 in length.')
  # Use the globally defined polynomial coefficients for the model.
  poly_coeffs = POLY_COEFFS
  poly_order = poly_coeffs.shape[0] - 1
  _, audf_powers = audio_tools.cf_to_audf(frequencies, poly_order)
  num_components = len(model_coeffs)
  components = audf_powers @ poly_coeffs[:, :num_components]
  # Calculate the dB HL values and reshape to a column vector to match Matlab.
  dbhl = (components @ model_coeffs).reshape(-1, 1)
  slopes = dbhl_to_slopes(frequencies, dbhl)
  return dbhl, slopes

def sones_subject_to_sones_nh(sones_subject: np.ndarray,
                              frequencies: np.ndarray,
                              subject_loudness_model: dict
                              ) -> np.ndarray:
  """Converts sones for a HI subject to equivalent sones for a normal listener.

  This function maps loudness values in sones from the hearing-impaired (HI)
  subject's perception to equivalent loudness values in sones for a
  normal-hearing (NH) listener

  Args:
      sones_subject: Sones for the subject.
      frequencies: Frequencies in Hz.
      subject_loudness_model: A dictionary containing the subject's loudness
                              model parameters.

  Returns:
      The equivalent sones for a normal hearing listener.
  """
  # Make sure frequencies are in a 1D format.
  frequencies = frequencies.ravel()
  _, slopes = hearing_level_model(frequencies,
                                  subject_loudness_model['component_coeffs'])
  sone_intersection = subject_loudness_model['sone_intersection']
  # Adjust for possibly higher-than-normal growth of loudness. For HI subjects
  # with slope > 1, the NH subject will sense a reduced range of level or
  # loudness, so exponent < 1.
  sones_nh = ((sones_subject / sone_intersection) **
              (1.0 / slopes)) * sone_intersection
  return sones_nh

def loudness_model_loss_function(
    frequencies: np.ndarray,
    amplitudes: np.ndarray,
    cus: np.ndarray,
    subject_loudness_model: dict
) -> float:
  """Calculates the loss for a loudness model.

  This function calculates the RMS error in dB between
  a) The dB SPL levels corresponding to the categorical units for a batch of
     test data, and
  b) The db SPL levels produced by the loudness model.

  Args:
      frequencies: The frequencies in Hz.
      amplitudes: The amplitudes of the signals.
      cus: The categorical units of loudness (CUs).
      subject_loudness_model: A dictionary containing the subject's loudness
                              model parameters.

  Returns:
      The loss value (root mean squared error in dB).
  """
  # Find valid categorical unit values (excluding "can't hear" and "too loud").
  # TODO: consider defining these constants elsewhere.
  valid_cus_indices = np.where((cus > 0) & (cus < 45))[0]
  if not valid_cus_indices.size:
    # TODO: consider whether we should return None rather than 0 here, since
    # 0 might indicate a perfect fit, whereas None would indicate no valid CUs.
    return 0.0  # No valid CUs, return 0 loss.
  else:
    # Filter data based on valid categorical units.
    frequencies = frequencies[valid_cus_indices]
    amplitudes = amplitudes[valid_cus_indices]
    cus = cus[valid_cus_indices]
    # Convert CUs to sones for the subject.
    sones_subject = audio_tools.cus_to_sones(cus)
    # Map subject's sones to equivalent sones for a normal hearing listener,
    # which would be a reduced range of loudness.
    # Make sure data is in 1D vector format, not a matrix.
    sones_nh = sones_subject_to_sones_nh(
        sones_subject, frequencies, subject_loudness_model
    )
    phons = audio_tools.sones_to_phons(sones_nh)
    dbspl = phons_to_dbspl(frequencies, phons)
    # Calculate the root mean squared error compared to the presented dB SPL.
    presented_dbspl = 20 * np.log10(amplitudes) + 94
    loss = np.sqrt(np.mean((dbspl - presented_dbspl) ** 2))  # RMS dB.
    return loss

def random_test_frequency_and_amplitude(
    min_freq: float,
    max_freq: float,
    loudness_model: dict,
) -> tuple[float, float]:
  """Generates a random test frequency and amplitude.

  Args:
      min_freq: The minimum frequency in Hz.
      max_freq: The maximum frequency in Hz.
      loudness_model: A dictionary containing the subject's loudness model
                      parameters.

  Returns:
      A tuple containing the generated frequency in Hz and its
      RMS amplitude.
  """

  min_cu = MIN_STIMULUS_CU
  max_cu = 45
  full_phon_range = 90

  ok = False
  phon_level = None
  while not ok:  # Try multiple times to generate a "good" random pair.
    frequency = np.exp(
        np.log(min_freq) +
        (np.log(max_freq) - np.log(min_freq)) * np.random.rand(1)
    )

    # Map top and bottom CUs for this frequency for the modeled subject:
    min_sones = sones_subject_to_sones_nh(
        audio_tools.cus_to_sones(min_cu), frequency, loudness_model
    )
    max_sones = sones_subject_to_sones_nh(
        audio_tools.cus_to_sones(max_cu), frequency, loudness_model
    )

    # phons are defined as physical units, nh-related only.
    min_phons = audio_tools.sones_to_phons(min_sones)
    max_phons = audio_tools.sones_to_phons(max_sones)

    # Don't let the range get too high or too small.
    max_phons = min(max_phons, 90)
    min_phons = min(min_phons, max_phons - 20)  # Make extra inaudible highs.

    if np.random.rand(
            1
    ) < (max_phons - min_phons) / full_phon_range:  # Prune where range small.
      ok = True
      phon_level = min_phons + (max_phons - min_phons) * np.random.rand(1)

  amplitude = calibration.dbspl_to_amp(phons_to_dbspl(frequency,
                                                      phon_level))

  return frequency.item(), amplitude.item()

def random_test_frequencies_and_amplitudes(
    min_freq: float,
    max_freq: float,
    n_samples: int,
    loudness_model: dict
) -> tuple[np.ndarray, np.ndarray]:
  """Generates random test frequencies and amplitudes.

  Args:
      min_freq: The minimum frequency in Hz.
      max_freq: The maximum frequency in Hz.
      n_samples: The number of samples to generate.
      loudness_model: A dictionary containing the subject's loudness model
                      parameters.

  Returns:
      A tuple containing the generated frequencies in Hz and their
      RMS amplitudes, as numpy column vectors of dimensions n_samples.
  """
  # Generate random frequencies on a log scale.
  frequencies = np.exp(
      np.log(min_freq) +
      (np.log(max_freq) - np.log(min_freq)) * np.random.rand(n_samples, 1))
  # Generate random amplitudes uniformly in phons or
  # log(sones) space, from MIN_STIMULUS_CU to max_cu.
  # Map top and bottom CUs for each frequency for the
  # modeled subject.
  min_cu = MIN_STIMULUS_CU
  min_sones = sones_subject_to_sones_nh(audio_tools.cus_to_sones(min_cu),
                                        frequencies, loudness_model)
  max_cu = 45
  max_sones = sones_subject_to_sones_nh(audio_tools.cus_to_sones(max_cu),
                                        frequencies, loudness_model)
  # Convert sones to phons (physical units - NH-related only).
  min_phons = audio_tools.sones_to_phons(min_sones)
  max_phons = audio_tools.sones_to_phons(max_sones)
  # Generate uniform random levels in phon space.
  phons = min_phons + (max_phons - min_phons) * np.random.rand(n_samples, 1)
  # Convert phons to dB SPL.
  dbspl = phons_to_dbspl(frequencies, phons)
  # Convert dB SPL to amplitude.
  # TODO(dicklyon): See if we need to calibrate a scale that's better.
  dbfs = 94.0  # Arbitrary full-scale scaling for now
  amplitudes = 10 ** ((dbspl - dbfs) / 20)

  return frequencies, amplitudes

def loudness_model_to_audiogram(loudness_model: dict) -> dict:
  """Converts a loudness model to an audiogram.

  Args:
      loudness_model: A dictionary containing the subject's loudness model
                      parameters.

  Returns:
      A dictionary containing the audiogram data, with keys 'frequencies' and
      'hearing_levels'.
  """
  frequencies = np.array(
    [250, 500, 1000, 1500, 2000, 3000, 4000, 6000, 8000]).reshape(-1, 1)
  # Finding hearing_levels, difference in dB between the 1/16 sone normal
  # curve and the 1/16 sone subject curve (1/16 sone is 0 phon) converted to
  # normal sones, phons, and dbspl.
  sones = 1/16
  sones_nh = sones_subject_to_sones_nh(sones, frequencies, loudness_model)
  dbspl_nominal = phons_to_dbspl(frequencies, audio_tools.sones_to_phons(sones))
  dbspl_actual = phons_to_dbspl(frequencies,
                                audio_tools.sones_to_phons(sones_nh))
  hearing_levels = dbspl_actual - dbspl_nominal
  audiogram = {
      'frequencies': frequencies,
      'hearing_levels': hearing_levels
  }
  return audiogram

def simulate_loudness_categorization(
    frequency: float,
    amplitude: float,
    loudness_model: dict,
    error_rate: float
) -> float:
  """Simulates a category response for a presented tone.

  Args:
      frequency: The frequency of the tone in Hz.
      amplitude: The amplitude of the tone.
      loudness_model: A dictionary containing the subject's loudness model
                      parameters.
      error_rate: The probability of the simulated response being incorrect.

  Returns:
      The simulated category response in categorical units of loudness (CUs).
  """

  # Reference data for category levels. Note that the "not heard" category
  # must still be positive.
  cus = BUTTONS_TO_CUS.reshape(-1, 1)
  sones_subject = audio_tools.cus_to_sones(cus)
  # Convert subject's sones to equivalent sones for a normal hearing listener.
  sones_nh = sones_subject_to_sones_nh(sones_subject, np.array([frequency]),
                                       loudness_model)
  phons = audio_tools.sones_to_phons(sones_nh)
  frequencies = np.ones(cus.shape) * frequency
  dbspl = phons_to_dbspl(frequencies, phons)
  # Convert dB SPL to amplitude, using a calibrated approach.
  presented_dbspl = calibration.amp_to_dbspl(amplitude)
  # Calculate distances to presented dB SPL.
  distances = (dbspl - presented_dbspl) ** 2
  # Pick the category with minimum distance, or not far from it.
  # TODO: this feels like a bit of a hack - replace with a direct selection
  # based on np.random.choice or similar.
  index = np.argmin(distances)
  for _ in range(2):
    if np.random.rand() > error_rate:
      break
    distances[index] = 10000
    index = np.argmin(distances)
  return cus[index][0]

def update_loudness_model(
    frequencies: np.ndarray,
    amplitudes: np.ndarray,
    cus: np.ndarray,
    loudness_model: dict,
    rate: float,
) -> tuple[dict, list[float]]:
  """Updates the subject's loudness model based on CLS test results.

  Args:
      frequencies: The frequencies in Hz.
      amplitudes: The RMS amplitudes of the signals relative to full scale.
      cus: The categorical units of loudness (CUs).
      loudness_model: A dictionary containing the subject's loudness model
                      parameters (optional).
      rate: The rate of adjustment towards the new optimum (optional).

  Returns:
      A tuple containing the updated loudness model and a list of loss
      values formatted as [old_loss, best_loss, updated_loss].
  """
  loudness_model = loudness_model.copy()  # Prevent side effects in the caller.
  old_loss = loudness_model_loss_function(
    frequencies, amplitudes, cus, loudness_model
  )
  component_coeffs = loudness_model['component_coeffs'].copy()
  sone_intersection = loudness_model['sone_intersection']

  # Grid search to optimize parameters.
  inc = 0.1  # Arbitrary good-enough grid increment.
  inc2 = 0.5  # For the sone_intersection, near 24.
  improved = True  # Go through the loop at least once.
  best_loss = old_loss
  max_iterations = 100 # Limit iterations to prevent potential hangs.
  iteration_count = 0
  while improved and iteration_count < max_iterations:
    iteration_count += 1
    improved = False
    for dim in range(len(component_coeffs)):
      coeffs = component_coeffs.copy()
      coeffs[dim] = component_coeffs[dim] + inc
      test_model = {'component_coeffs': coeffs.copy(),
                    'sone_intersection': sone_intersection}
      new_loss = loudness_model_loss_function(frequencies, amplitudes, cus,
                                              test_model)
      if new_loss < best_loss:
        best_loss = new_loss
        improved = True
        component_coeffs = coeffs.copy()
      # Also try the negative increment.
      coeffs[dim] = component_coeffs[dim] - inc
      test_model = {'component_coeffs': coeffs.copy(),
                    'sone_intersection': sone_intersection}
      new_loss = loudness_model_loss_function(frequencies, amplitudes, cus,
                                              test_model)
      if new_loss < best_loss:
        best_loss = new_loss
        improved = True
        component_coeffs = coeffs.copy()

    # Search on sone_intersection dimension here.
    test_model = {'component_coeffs': component_coeffs.copy(),
                  'sone_intersection': sone_intersection + inc2}
    new_loss = loudness_model_loss_function(frequencies, amplitudes, cus,
                                            test_model)
    if new_loss < best_loss:
      best_loss = new_loss
      improved = True
      sone_intersection = sone_intersection + inc2

    test_model = {'component_coeffs': component_coeffs.copy(),
                  'sone_intersection': sone_intersection - inc2}
    new_loss = loudness_model_loss_function(frequencies, amplitudes, cus,
                                            test_model)
    if new_loss < best_loss:
      best_loss = new_loss
      improved = True
      sone_intersection = sone_intersection - inc2

  # Check if the loop terminated due to max iterations.
  if iteration_count >= max_iterations:
    print(f'Warning: Loudness model update reached max iterations '
          f'({max_iterations}) without full convergence.')
  else:
    print(f'Model fitting took {iteration_count} iterations')

  # Update the model partially toward the new optimum by an amount
  # 0 < rate < 1.
  new_coeffs = loudness_model['component_coeffs'] + rate * (
      component_coeffs - loudness_model['component_coeffs'])
  loudness_model['component_coeffs'] = new_coeffs
  new_sone_int = loudness_model['sone_intersection'] + rate * (
      sone_intersection - loudness_model['sone_intersection'])
  loudness_model['sone_intersection'] = new_sone_int
  updated_loss = loudness_model_loss_function(frequencies, amplitudes, cus,
                                              loudness_model)
  losses = [old_loss, best_loss, updated_loss]
  return loudness_model, losses
