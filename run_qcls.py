"""Runs the qCLS process on simulated data."""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import calibration
import hearing_models

def save_csv(freq_hz, amp, buttons, filename):
  """Save the data to a CSV file."""
  data = np.column_stack([freq_hz, amp, buttons])
  df = pd.DataFrame(data, columns=['Frequency (Hz)', 'Amplitude', 'Button'])
  # Save the button-press column as integer type.
  df['Button'] = df['Button'].astype(int)
  # Add index column and call it 'Stimulus'
  df.to_csv(filename, index_label='Stimulus')

def qcls_testing_process(hidden_model: dict = None, error_rate: float = 0.33,
                         batch_size: int = 20, num_batches: int = 5,
) -> tuple[dict, float]:
  """Runs the qCLS process on simulated data.

  Args:
    hidden_model: A dictionary representing the hidden loudness model.
                  Defaults to a mild-moderate hearing loss model.
    error_rate: The probability of the simulated user making an error.
    batch_size: The number of trials per batch.
    num_batches: The total number of batches.

  Returns:
    A tuple containing the updated loudness model and RMS error between the
    hidden and inferred audiogram.
  """
  if hidden_model is None:
    hidden_model = {
        'component_coeffs': np.asarray([1, 0, 0]),  # Mean mild-moderate.
        'sone_intersection': 24
    }
  # Start by assuming the average mild-to-moderate loss:
  loudness_model = {
      'component_coeffs': np.asarray([1, 0]),  # Average mild-moderate loss.
      'sone_intersection': 24
  }
  # First batch on the most important range.
  min_freq = 1000
  max_freq = 4000
  all_data = np.zeros((batch_size * num_batches, 3))
  start_idx = 0
  for batch in range(1, num_batches + 1):
    frequencies, amps = hearing_models.random_test_frequencies_and_amplitudes(
        min_freq, max_freq, batch_size, loudness_model)

    # Simulate user responses (vectorized).
    cus = np.array([
        hearing_models.simulate_loudness_categorization(
            f, a, hidden_model, error_rate
        )
        for f, a in zip(frequencies, amps)
    ])
    cus = cus.reshape(-1, 1)  # Convert to column vector.

    end_idx = start_idx + batch_size
    all_data[start_idx:end_idx, :] = np.column_stack(
        [frequencies, amps, cus]
    )
    start_idx = end_idx

    # Update the model.
    learning_rate = 1.25 / (batch + 1)
    new_loudness_model, _ = hearing_models.update_loudness_model(
        frequencies, amps, cus, loudness_model, learning_rate)

    loudness_model = new_loudness_model.copy()
    # Expand the range for subsequent batches.
    min_freq = 250
    max_freq = 8000

  # Update based on all data.
  frequencies = all_data[:, 0].reshape(-1, 1)
  amplitudes = all_data[:, 1].reshape(-1, 1)
  cus = all_data[:, 2].reshape(-1, 1)
  # Save the data to a CSV file in local_results.
  os.makedirs('local_results', exist_ok=True)
  buttons = [np.where(hearing_models.BUTTONS_TO_CUS == cu)[0][0] for cu in cus]
  save_csv(frequencies, amplitudes, buttons, 'local_results/simulated_data.csv')

  plt.figure()
  plt.semilogx(frequencies, calibration.amp_to_dbspl(amplitudes), 'r*')
  plt.title('Simulated user response CU values')
  plt.xlabel('Frequency, Hz')
  plt.ylabel('dB SPL presented')
  for n in range(len(all_data)):
    freq_hz = frequencies[n].item()
    amp_dbspl = calibration.amp_to_dbspl(amplitudes[n].item())
    plot_str =  str(int(cus[n].item()))
    plt.text(freq_hz, amp_dbspl, plot_str)

  learning_rate = 0.5
  loudness_model, _ = hearing_models.update_loudness_model(
      frequencies, amplitudes, cus, loudness_model, learning_rate)

  hidden_audiogram = hearing_models.loudness_model_to_audiogram(hidden_model)
  inf_audiogram = hearing_models.loudness_model_to_audiogram(loudness_model)

  plt.semilogx(hidden_audiogram['frequencies'],
               hidden_audiogram['hearing_levels'],
               'bo-', label='Hidden Audiogram')
  plt.semilogx(inf_audiogram['frequencies'],
               inf_audiogram['hearing_levels'],
               'rs-', label='Inferred Threshold')
  plt.legend()

  audigram_errors_db = (np.array(hidden_audiogram['hearing_levels']) -
                        np.array(inf_audiogram['hearing_levels']))
  rms_audiogram_error = np.sqrt(np.mean(audigram_errors_db**2))
  plt.show()
  return loudness_model, rms_audiogram_error

def plot_qcls_results_mono(df: pd.DataFrame, loudness_model: dict):
  """Plots the qCLS results for mono data.

  Args:
    df: The DataFrame containing the saved data with column headers,
        'Frequency (Hz)', 'Amplitude', and 'Button'.
    loudness_model: The calculated loudness model.

  Returns:
    matplotlib.figure: The figure containing the plot.
  """
  frequencies = df['Frequency (Hz)'].values.reshape(-1, 1)
  amplitudes = df['Amplitude'].values.reshape(-1, 1)
  buttons = df['Button'].values
  cus = hearing_models.BUTTONS_TO_CUS[buttons].reshape(-1, 1)

  fig = plt.figure(figsize=(8, 5))
  plt.title('Responses with fitted model')
  plt.xlabel('Frequency, Hz')
  plt.ylabel('dB SPL presented')

  # Plot points with color-coding based on button press index (0 to 10)
  sc = plt.scatter(
      frequencies,
      calibration.amp_to_dbspl(amplitudes),
      c=buttons,
      cmap='viridis',
      marker='*',
      s=100,
      vmin=0,
      vmax=10,
      zorder=3,
  )
  plt.xscale('log')
  plt.colorbar(sc, label='Button Press (0-10)')

  for n in range(len(frequencies)):
    freq_hz = frequencies[n].item()
    amp_dbspl = calibration.amp_to_dbspl(amplitudes[n].item())
    plot_str = str(int(cus[n].item()))
    plt.text(freq_hz, amp_dbspl, plot_str)

  inf_audiogram = hearing_models.loudness_model_to_audiogram(loudness_model)

  plt.semilogx(inf_audiogram['frequencies'],
               inf_audiogram['hearing_levels'],
               'rs-',
               label='Inferred Threshold')
  plt.legend()
  plt.grid(axis='y')
  return fig

def plot_qcls_results_stereo(df_left: pd.DataFrame,
                             loudness_model_left: dict,
                             df_right: pd.DataFrame,
                              loudness_model_right: dict):
  """Plots the qCLS results for stereo data.

  Args:
    df_left: DataFrame containing the left ear data with columns
            'Frequency (Hz)', 'Amplitude', and 'Button'.
    loudness_model_left: The calculated loudness model for left ear.
    df_right: DataFrame containing the right ear data.
    loudness_model_right: The calculated loudness model for right ear.

  Returns:
    matplotlib.figure: The figure containing the stereo plots.
  """
  # Process left ear data.
  frequencies_left = df_left['Frequency (Hz)'].values.reshape(-1, 1)
  amplitudes_left = df_left['Amplitude'].values.reshape(-1, 1)
  buttons_left = df_left['Button'].values
  cus_left = hearing_models.BUTTONS_TO_CUS[buttons_left].reshape(-1, 1)
  # Process right ear data.
  frequencies_right = df_right['Frequency (Hz)'].values.reshape(-1, 1)
  amplitudes_right = df_right['Amplitude'].values.reshape(-1, 1)
  buttons_right = df_right['Button'].values
  cus_right = hearing_models.BUTTONS_TO_CUS[buttons_right].reshape(-1, 1)
  # Set up subplots.
  fig, ax = plt.subplots(1, 2, sharey=True, figsize=(14, 7))
  font = 20
  # Left ear subplot.
  ax[0].set_title('Left Ear', size=font)
  ax[0].set_xlabel('Frequency, Hz', size=font)
  ax[0].set_ylabel('dB SPL presented', size=font)
  for n in range(len(frequencies_left)):
    freq_hz = frequencies_left[n].item()
    amp_dbspl = calibration.amp_to_dbspl(amplitudes_left[n].item())
    plot_str = str(int(cus_left[n].item()))
    ax[0].semilogx(freq_hz, amp_dbspl, 'b*')
    ax[0].text(freq_hz, amp_dbspl, plot_str)
  # Plot left ear audiogram.
  inf_audiogram_left = hearing_models.loudness_model_to_audiogram(
    loudness_model_left)
  ax[0].semilogx(inf_audiogram_left['frequencies'],
                   inf_audiogram_left['hearing_levels'],
                   'bs-',
                   label='Inferred Threshold')
  ax[0].legend()
  ax[0].grid(axis='y')
  # Right ear subplot.
  ax[1].set_title('Right Ear', size=font)
  ax[1].set_xlabel('Frequency, Hz', size=font)
  ax[1].set_ylabel('dB SPL presented', size=font)
  for n in range(len(frequencies_right)):
    freq_hz = frequencies_right[n].item()
    amp_dbspl = calibration.amp_to_dbspl(amplitudes_right[n].item())
    plot_str = str(int(cus_right[n].item()))
    ax[1].semilogx(freq_hz, amp_dbspl, 'r*')
    ax[1].text(freq_hz, amp_dbspl, plot_str)
  # Plot right ear audiogram.
  inf_audiogram_right = hearing_models.loudness_model_to_audiogram(
    loudness_model_right)
  ax[1].semilogx(inf_audiogram_right['frequencies'],
                    inf_audiogram_right['hearing_levels'],
                    'rs-',
                    label='Inferred Threshold')
  ax[1].legend()
  ax[1].grid(axis='y')
  plt.tight_layout()
  return fig


def run_on_complete_dataset(df: pd.DataFrame) -> dict:
  """Runs the qCLS process on a complete set of collected data.

  This function calculates the loudness model based on the provided data.

  Args:
    df: The DataFrame containing the saved data with column headers,
        'Ear', 'Frequency (Hz)', 'Amplitude', and 'Button'.

  Returns:
    The loudness model (dict).
  """
  # A small non-zero starting point is used to avoid numerical/gradient
  # degeneracy and convergence issues that can occur if initialized at
  # exactly [0, 0].
  loudness_model = {
    'component_coeffs': np.asarray([0.01, 0]),
    'sone_intersection': 24
  }
  frequencies = df['Frequency (Hz)'].values.reshape(-1, 1)
  amplitudes = df['Amplitude'].values.reshape(-1, 1)
  buttons = df['Button'].values
  cus = hearing_models.BUTTONS_TO_CUS[buttons].reshape(-1, 1)
  learning_rate = 1
  loudness_model, _ = hearing_models.update_loudness_model(
    frequencies, amplitudes, cus, loudness_model, learning_rate)
  return loudness_model

def main():
  parser = argparse.ArgumentParser(
    description='Run the qCLS process on simulated data.')
  parser.add_argument('--csv_file', type=str, help='CSV file to use.')
  args = parser.parse_args()
  if args.csv_file:
    df = pd.read_csv(args.csv_file)
    model = run_on_complete_dataset(df)
    print(model)
    audiogram_plt = plot_qcls_results_mono(df, model)
    audiogram_plt.show()
  else:
    qcls_testing_process()

if __name__ == '__main__':
  main()  # pragma: no cover

