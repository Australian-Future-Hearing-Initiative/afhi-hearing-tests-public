"""Comparison tests between Python and MATLAB code, to help with porting."""

import inspect
import matlab.engine
import numpy as np

import audio_tools
import hearing_models


def dbspl_to_amplitude_test(eng):
  """Test the dbspl_to_amplitude function in Python and MATLAB."""
  test_cases = [
      (94.0, 1.0),
      (100.0, 1.0),
      (94.0, 0.1),
  ]
  for fs_db, amplitude in test_cases:
    pyt = audio_tools.dbspl_to_amplitude(fs_db, amplitude)
    mat = eng.dbspl_to_amplitude(fs_db, amplitude)
    report_results(pyt, mat)

def amplitude_to_dbspl_test(eng):
  """Test the amplitude_to_dbspl function in Python and MATLAB."""
  test_cases = [
      (1.0, 94.0),
      (0.1, 94.0),
      (1.0, 100.0),
  ]
  for amplitude, fs_db in test_cases:
    pyt = audio_tools.amplitude_to_dbspl(amplitude, fs_db)
    mat = eng.amplitude_to_dbspl(amplitude, fs_db)
    report_results(pyt, mat)

def cus_to_sones_test(eng):
  """Test the cus_to_sones function in Python and MATLAB."""
  test_cases = [1.0, 42.0]
  for cus in test_cases:
    pyt = audio_tools.cus_to_sones(cus)
    mat = eng.cus_to_sones(cus)
    report_results(pyt, mat)

def sones_to_cus_test(eng):
  """Test the sones_to_cus function in Python and MATLAB."""
  test_cases = [1.0, 42.0]
  for sones in test_cases:
    pyt = audio_tools.sones_to_cus(sones)
    mat = eng.sones_to_cus(sones)
    report_results(pyt, mat)

def sones_to_phons_test(eng):
  """Test the sones_to_phons function in Python and MATLAB."""
  test_cases = [1.0, 60.0]
  for sones in test_cases:
    pyt = audio_tools.sones_to_phons(sones)
    mat = eng.sones_to_phons(sones)
    report_results(pyt, mat)

def phons_to_sones_test(eng):
  """Test the phons_to_sones function in Python and MATLAB."""
  test_cases = [40.0, 60.0]
  for phons in test_cases:
    pyt = audio_tools.phons_to_sones(phons)
    mat = eng.phons_to_sones(phons)
    report_results(pyt, mat)

def cf_to_audf_test(eng):
  """Test the cf_to_audf function in Python and MATLAB."""
  cf = np.array([1000.0, 2000.0])
  pyt, pyt_powers = audio_tools.cf_to_audf(cf)
  mat, mat_powers = eng.CF_to_audf(cf, nargout=2)
  report_results(pyt, mat)
  report_results(pyt_powers, mat_powers)

def phons_to_dbspl_test(eng):
  """Test the phons_to_dbspl function in Python and MATLAB."""
  # Try with a scalar phon level.
  freq_hz = np.array([1000.0, 2000.0]).reshape(-1, 1)
  phon_level = 40.0
  pyt = hearing_models.phons_to_dbspl(freq_hz, phon_level)
  mat = eng.phons_to_dbspl(freq_hz, phon_level)
  report_results(pyt, mat)
  # Now try with an array of phon levels.
  phon_level = np.array([40.0, 60.0]).reshape(-1, 1)
  pyt = hearing_models.phons_to_dbspl(freq_hz, phon_level)
  mat = eng.phons_to_dbspl(freq_hz, phon_level)
  report_results(pyt, mat)

def dbhl_to_slopes_test(eng):
  """Test the dbhl_to_slopes function in Python and MATLAB."""
  frequency = np.array([1000.0, 2000.0, 4000.0])
  dbhl = np.array([20.0, 30.0, 40.0])
  pyt = hearing_models.dbhl_to_slopes(frequency, dbhl)
  mat = eng.dbhl_to_slopes(frequency, dbhl)
  report_results(pyt, mat)

def hearing_level_model_test(eng):
  """Test the hearing_level_model function in Python and MATLAB."""
  freqs_hz = np.array([1000.0, 2000.0, 4000.0])
  coeffs = np.array([0.5, 1.0, -1.0, 1.0])
  py_dbhl, py_slopes = hearing_models.hearing_level_model(freqs_hz, coeffs)
  mat_dbhl, mat_slopes = eng.hearing_level_model(freqs_hz, coeffs, nargout=2)
  report_results(py_dbhl, mat_dbhl)
  report_results(py_slopes, mat_slopes)

def sones_subject_to_sones_nh_test(eng):
  """Test the sones_subject_to_sones_nh function in Python and MATLAB."""
  sones = np.array([1.0, 20.0, 30.0]).reshape(-1, 1)
  freqs_hz = np.array([1000.0, 2000.0, 4000.0]).reshape(-1, 1)
  model = {'component_coeffs': np.asarray([1.0, 0.0]),
           'sone_intersection': 24.0}
  py = hearing_models.sones_subject_to_sones_nh(sones, freqs_hz, model)
  mat = eng.sones_subject_to_sones_nh(sones, freqs_hz, model)
  report_results(py, mat)

def loudness_model_loss_function_test(eng):
  """Test the loudness_model_loss_function function in Python and MATLAB."""
  freqs_hz = np.array([1000.0, 2000.0, 4000.0]).reshape(-1, 1)
  amps = np.array([0.5, 0.6, 0.7]).reshape(-1, 1)
  cus = np.array([30.0, 25.0, 20.0]).reshape(-1, 1)
  model = {'component_coeffs': np.asarray([1.0, 0.0]),
           'sone_intersection': 24.0}
  py = hearing_models.loudness_model_loss_function(freqs_hz, amps, cus, model)
  mat = eng.loudness_model_loss_function(freqs_hz, amps, cus, model)
  report_results(py, mat)

def random_test_frequency_and_amplitude_test(eng):
  """Test random_test_frequency_and_amplitude in Python and Matlab."""
  min_freq = 1000.0
  max_freq = 6000.0
  loudness_model = {'component_coeffs': np.array([1.0, 0.0]).reshape(-1, 1),
                    'sone_intersection': 24.0}
  py_freq, py_amp = hearing_models.random_test_frequency_and_amplitude(
    min_freq, max_freq, loudness_model)
  mat_freq, mat_amp = eng.random_test_frequency_and_amplitude(
    min_freq, max_freq, loudness_model, nargout=2)
  assert isinstance(py_freq, float)
  assert isinstance(py_amp, float)
  assert isinstance(mat_freq, float)
  assert isinstance(mat_amp, float)
  print('Assertion passed in random_test_frequency_and_amplitude()')

def random_test_frequencies_and_amplitudes_test(eng):
  """Test random_test_frequencies_and_amplitudes in Python and Matlab."""
  # Actual numbers produced will be random, so just check dimensions.
  min_freq = 1000.0
  max_freq = 6000.0
  n_samples = 10
  loudness_model = {'component_coeffs': np.array([1.0, 0.0]),
                    'sone_intersection': 24.0}
  py_freqs, py_amp = hearing_models.random_test_frequencies_and_amplitudes(
    min_freq, max_freq, n_samples, loudness_model)
  mat_freqs, mat_amp = eng.random_test_frequencies_and_amplitudes(
    min_freq, max_freq, n_samples, loudness_model, nargout=2)
  report_results(py_freqs.shape, mat_freqs.size)
  report_results(py_amp.shape, mat_amp.size)

def loudness_model_to_audiogram_test(eng):
  """Test the loudness_model_to_audiogram function in Python and MATLAB."""
  model = {'component_coeffs': np.asarray([1.0, 0.0]),
           'sone_intersection': 24.0}
  py = hearing_models.loudness_model_to_audiogram(model)
  mat = eng.loudness_model_to_audiogram(model)
  report_results(py['frequencies'], mat['frequencies'])
  report_results(py['hearing_levels'], mat['hearing_levels'])

def simulate_loudness_categorization_test(eng):
  """Test the simulate_loudness_categorization function in Python and MATLAB."""
  freq_hz = 1000.0
  amp = 0.5
  model = {'component_coeffs': np.asarray([1.0, 0.0]),
           'sone_intersection': 24.0}
  error_rate = 0  # Use zero to ensure repeatable results.
  py = hearing_models.simulate_loudness_categorization(
    freq_hz, amp, model, error_rate)
  mat = eng.simulate_loudness_categorization(freq_hz, amp, model, error_rate)
  report_results(py, mat)

def update_loudness_model_test(eng):
  """Test the update_loudness_model function in Python and MATLAB."""
  frequencies = np.array([1000.0, 2000.0, 4000.0]).reshape(-1, 1)
  amplitudes = np.array([0.5, 0.6, 0.7]).reshape(-1, 1)
  cus = np.ndarray = np.array([30.0, 25.0, 20.0]).reshape(-1, 1)
  model = {'component_coeffs': np.asarray([1.0, 0.0, 0.0]),
           'sone_intersection': 24.0}
  rate = 0.5
  py_model, py_loss = hearing_models.update_loudness_model(
    frequencies, amplitudes, cus, model, rate)
  mat_model, mat_loss = eng.update_loudness_model(
    frequencies, amplitudes, cus, model, rate, nargout=2)
  report_results(py_model['component_coeffs'], mat_model['component_coeffs'])
  report_results(py_model['sone_intersection'], mat_model['sone_intersection'])
  report_results(py_loss, mat_loss)

def report_results(py, mat):
  """Compare output and report the results."""
  caller_name = inspect.stack()[1].function
  try:
    equality = np.allclose(py, mat, rtol=1e-5, atol=1e-5)
  except ValueError:
    # The comparison raised an exception, which likely means that the format
    # of the data wasn't compatible, which is certainly a fail.
    equality = False
  if equality:
    print(f'Assertion passed in {caller_name}()')
  else:
    print(f'Assertion FAILED in {caller_name}()')
    print('---- Python output ----')
    print(py)
    print(type(py))
    print('---- MATLAB output ----')
    print(mat)
    print(type(mat))

def main():
  # Set up the Matlab engine.
  engine = matlab.engine.connect_matlab()
  engine.addpath('reference_matlab_code')
  # Run the tests.
  test_functions = [dbspl_to_amplitude_test,
                    amplitude_to_dbspl_test,
                    cus_to_sones_test,
                    sones_to_cus_test,
                    sones_to_phons_test,
                    phons_to_sones_test,
                    cf_to_audf_test,
                    phons_to_dbspl_test,
                    random_test_frequency_and_amplitude_test,
                    random_test_frequencies_and_amplitudes_test,
                    dbhl_to_slopes_test,
                    hearing_level_model_test,
                    sones_subject_to_sones_nh_test,
                    loudness_model_loss_function_test,
                    loudness_model_to_audiogram_test,
                    simulate_loudness_categorization_test,
                    update_loudness_model_test,
                    ]
  for test_function in test_functions:
    test_function(engine)
  # Disconnect from the engine.
  engine.quit()

if __name__ == '__main__':
  main()
