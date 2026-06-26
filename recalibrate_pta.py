"""Functions to analyze and plot pure tone audiometry calibration data.

This script is designed to be run from the command line to calculate new
calibration values for pure-tone audiometry (PTA) tests. It takes a directory
of calibration data files and an existing calibration file to compute an
updated calibration.

The script expects a directory containing:
1. A directory containing one or more CSV files, each with data for a single
   subject. The expected CSV file format requires the following columns:
   - 'Frequency_hz': The test frequencies.
   - 'Ground_truth_left': Audiologist-measured threshold for the left ear.
   - 'Ground_truth_right': Audiologist-measured threshold for the right ear.
   - 'HW_left', 'HW_right': Test results for the Hughson-Westlake method.
   - 'Adaptive_left', 'Adaptive_right': Test results for the Adaptive method.
   - 'Adaptive_retest_left', 'Adaptive_retest_right': Retest results for the
     Adaptive method (optional).

2. An existing calibration file, which is used as a baseline. The script will
   look for either 'existing_calibration_hw.csv' or
   'existing_calibration_adaptive.csv', depending on the method specified.
   The script will first look in the data directory, and if it is not found
   there, it will look in the parent directory.
   The format for these files is:
   - 'Frequency (Hz)': The test frequencies.
   - 'Calibration': The existing calibration values in dB.

Usage:
  python recalibrate_pta.py path/to/your/data_files --method <HW|Adaptive>
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import scipy.ndimage


OUTLIER_STD_THRESHOLD = 2  # Number of std deviations to define an outlier.
COLOR_OLD_CAL = '#1f77b4'  # Muted blue.
COLOR_NEW_CAL = '#ff9b42'  # Bright, light orange.
COLOR_FILTERED_CAL = '#d26900'  # Dark, saturated orange.


def load_data(file_path: str) -> pd.DataFrame:
  """Loads re-calibration data from a CSV file into a DataFrame.

  Args:
      file_path (str): Path to the CSV file.

  Returns:
      pd.DataFrame: A DataFrame containing the calibration data.
  """
  # Read the CSV file into a DataFrame, using the first row as the header.
  df = pd.read_csv(file_path, header=0)
  return df


def _load_subject_data(dir_path: str) -> tuple[list[pd.DataFrame], list[str]]:
  """Loads all subject calibration CSVs from a directory."""
  if not (os.path.exists(dir_path) and os.path.isdir(dir_path)):
    raise FileNotFoundError(f'Data directory not found: {dir_path}')

  print(f'Looking for CSV files in: {dir_path}')
  calibration_dataframes = []
  calibration_filenames = []
  # These columns are essential for the script to run.
  required_cols = [
      'Frequency_hz', 'Ground_truth_left', 'Ground_truth_right',
      'HW_left', 'HW_right', 'Adaptive_left', 'Adaptive_right'
  ]

  for item_name in os.listdir(dir_path):
    file_path = os.path.join(dir_path, item_name)
    if (not item_name.lower().endswith('.csv') or
        not os.path.isfile(file_path) or
        'existing_calibration' in item_name.lower()):
      continue

    print(f'  Loading data from: {item_name}')
    df = pd.read_csv(file_path, header=0)

    if not all(col in df.columns for col in required_cols):
      raise ValueError(
        f'File {item_name} is missing one or more required columns.'
      )
    calibration_dataframes.append(df)
    calibration_filenames.append(item_name)
  return calibration_dataframes, calibration_filenames


def _load_existing_calibration(dir_path: str,
                               method: str) -> tuple[dict[int, float], str]:
  """Finds and loads the existing calibration file."""
  if method == 'HW':
    existing_cal_file_name = 'existing_calibration_hw.csv'
    method_name_for_print = 'Hughson-Westlake'
  else:  # 'Adaptive'
    existing_cal_file_name = 'existing_calibration_adaptive.csv'
    method_name_for_print = 'Adaptive'

  # Search for the calibration file in the data dir, then the parent directory.
  local_cal_path = os.path.join(dir_path, existing_cal_file_name)
  parent_dir = os.path.dirname(dir_path.rstrip('/\\'))
  parent_cal_path = os.path.join(parent_dir, existing_cal_file_name)

  existing_cal_path = None
  if os.path.isfile(local_cal_path):
    existing_cal_path = local_cal_path
    print(f'Found calibration file in data directory: {existing_cal_path}')
  elif os.path.isfile(parent_cal_path):
    existing_cal_path = parent_cal_path
    print('Found calibration file in parent directory: '
          f'{os.path.abspath(existing_cal_path)}')

  if existing_cal_path is None:
    raise FileNotFoundError(f'Could not find {existing_cal_file_name} in '
                            f'{dir_path} or its parent directory.')

  # Load and process the found calibration file.
  existing_df = load_data(existing_cal_path)
  if existing_df.empty:
    raise ValueError(
        f'Could not load data from {existing_cal_file_name} or it is empty.')
  try:
    # Convert the DataFrame to the desired dictionary format.
    existing_calibration_dict = {}
    for _, row in existing_df.iterrows():
      frequency = row['Frequency (Hz)']
      calibration_value = row['Calibration']
      freq_key = int(frequency)
      existing_calibration_dict[freq_key] = float(calibration_value)
    print(f'Successfully loaded and processed {existing_cal_file_name}.')
    return existing_calibration_dict, method_name_for_print
  except KeyError as e:
    raise KeyError(f'Missing column in {existing_cal_file_name}: {e}') from e


def process_calibration_directory(dir_path: str, method: str):
  """Processes a directory of calibration CSV files."""
  print(f'Processing directory for {method} method: {dir_path}')
  (calibration_dataframes,
   calibration_filenames) = _load_subject_data(dir_path)

  (existing_calibration_dict,
   method_name_for_print) = _load_existing_calibration(dir_path, method)

  print('\n--- Loading Summary ---')
  print(f'Loaded {len(calibration_dataframes)} data file(s)'
        f' from {dir_path}:')
  if calibration_filenames:
    for fname in calibration_filenames:
      print(f'  - {fname}')
  else:
    print('  (No files loaded)')
  print('\nExisting Calibration Dictionary:')
  if existing_calibration_dict:
    print(existing_calibration_dict)
  else:
    print('  (Not loaded or empty)')
  print('--- End of Summary ---\n')

  # Preprocess the dataframes: creates deep copies and adds mean columns.
  preprocessed_dfs = _preprocess_dataframes(calibration_dataframes, method)

  # Now calculate the new calibration, based on the existing calibration and
  # all the calibration data for the selected method.
  new_cal = calculate_version2_calibration(
      preprocessed_dfs,
      calibration_filenames,
      existing_calibration_dict,
      method
  )

  # Apply FIR filter to the new calibration.
  if new_cal:
    filtered_cal = _apply_fir_filter(new_cal)
  else:
    filtered_cal = None

  # Plot a comparison of the old and new calibrations.
  if new_cal and existing_calibration_dict:
    plot_calibration_comparison(existing_calibration_dict,
                                new_cal,
                                method_name_for_print,
                                filtered_cal)

  # Display a summary table of all calibration values.
  if new_cal and existing_calibration_dict:
    _display_calibration_table(existing_calibration_dict, new_cal, filtered_cal)

  # Generate a final plot of the mean absolute error by frequency, which also
  # calculates and reports the overall MAE for each condition.
  if new_cal and preprocessed_dfs and existing_calibration_dict:
    left_col, right_col = ('HW_left', 'HW_right') if method == 'HW' else \
                          ('Adaptive_mean_left', 'Adaptive_mean_right')
    plot_error_by_frequency(preprocessed_dfs, existing_calibration_dict,
                            new_cal, filtered_cal, method_name_for_print,
                            left_col, right_col)

  print('\n--- Diagnostics ---')
  num_subjects_used = len(calibration_dataframes)
  print(f'Number of subjects used in calibration: {num_subjects_used}')
  if new_cal:
    cal_values = list(new_cal.values())
    if cal_values:
      cal_range = max(cal_values) - min(cal_values)
      print(f'Calibration range (max - min): {cal_range:.2f} dB')
  print('--- End of Script ---')


def _preprocess_dataframes(dataframes: list[pd.DataFrame], method: str
                           ) -> list[pd.DataFrame]:
  """Creates deep copies of dataframes and adds mean columns for Adaptive.
  
  This function takes a list of dataframes and returns a new list of
  dataframes that have been preprocessed. For the 'Adaptive' method, it
  calculates the mean of the test and retest results.

  Args:
    dataframes: A list of Pandas DataFrames to be processed.
    method: The calibration method ('HW' or 'Adaptive').

  Returns:
    A new list of preprocessed Pandas DataFrames.
  """
  # Create deep copies to avoid any side effects on the original dataframes.
  processed_dfs = [df.copy(deep=True) for df in dataframes]

  if method == 'Adaptive':
    for df in processed_dfs:
      # Calculate the mean for the left ear.
      if 'Adaptive_retest_left' in df.columns:
        df['Adaptive_mean_left'] = df[['Adaptive_left',
                                       'Adaptive_retest_left']].mean(axis=1)
      elif 'Adaptive_left' in df.columns:
        df['Adaptive_mean_left'] = df['Adaptive_left']
      # Calculate the mean for the right ear.
      if 'Adaptive_retest_right' in df.columns:
        df['Adaptive_mean_right'] = df[['Adaptive_right',
                                        'Adaptive_retest_right']].mean(axis=1)
      elif 'Adaptive_right' in df.columns:
        df['Adaptive_mean_right'] = df['Adaptive_right']
  return processed_dfs


def plot_calibration_comparison(old_cal: dict[int, float],
                                new_cal: dict[int, float],
                                method_name: str,
                                filtered_cal: dict[int, float] = None):
  """Plots a comparison of old and new calibration values.

  Args:
    old_cal: Dictionary of the old calibration values.
    new_cal: Dictionary of the new calibration values.
    method_name: The name of the method for the plot title.
    filtered_cal: Optional dictionary of filtered calibration values.
  """
  _, ax = plt.subplots(figsize=(12, 6))
  # Ensure frequencies are sorted for plotting.
  freqs = sorted(old_cal.keys())
  old_vals = [old_cal[f] for f in freqs]
  new_vals = [new_cal[f] for f in freqs]
  ax.plot(freqs, old_vals, marker='o', linestyle='--', label='Old Calibration',
          color=COLOR_OLD_CAL)
  ax.plot(freqs, new_vals, marker='x', linestyle='--',
          label='New Calibration', color=COLOR_NEW_CAL)
  if filtered_cal:
    filtered_vals = [filtered_cal[f] for f in freqs]
    ax.plot(freqs, filtered_vals, marker='', linestyle='-',
            label='Filtered Calibration', color=COLOR_FILTERED_CAL, linewidth=2)
  ax.set_title(f'Calibration Comparison ({method_name})', fontsize=16)
  ax.set_xlabel('Frequency (Hz)', fontsize=12)
  ax.set_ylabel('Calibration Offset (dB)', fontsize=12)
  ax.set_xscale('log')
  ax.set_xticks(freqs)
  ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
  ax.legend()
  ax.grid(True, which='both', ls='--', alpha=0.6)
  plt.tight_layout()
  plt.show()


def _display_calibration_table(old_cal: dict[int, float],
                               new_cal: dict[int, float],
                               filtered_cal: dict[int, float] | None):
  """Displays a summary table of calibration values.

  Args:
    old_cal: Dictionary of the old calibration values.
    new_cal: Dictionary of the new calibration values.
    filtered_cal: Optional dictionary of filtered calibration values.
  """
  if not new_cal:
    return

  freqs = sorted(new_cal.keys())
  data = {
      'Old Calibration': [old_cal.get(f) for f in freqs],
      'New Calibration': [new_cal.get(f) for f in freqs],
  }
  if filtered_cal:
    data['Filtered Calibration'] = [filtered_cal.get(f) for f in freqs]

  df = pd.DataFrame(data, index=freqs)
  df.index.name = 'Frequency (Hz)'
  df = df.round(1)

  print('\n--- Calibration Summary Table ---')
  print(df.to_csv())


def _calculate_mae(dataframes: list[pd.DataFrame],
                     left_col: str, right_col: str) -> float:
  """Calculates the overall mean absolute error across all subjects and ears.

  Args:
    dataframes: A list of Pandas DataFrames with the hearing test results.
    left_col: The name of the column with the left ear test results.
    right_col: The name of the column with the right ear test results.

  Returns:
    The overall mean absolute error in dB.
  """
  all_errors = []
  for df in dataframes:
    errors_left = df[left_col] - df['Ground_truth_left']
    errors_right = df[right_col] - df['Ground_truth_right']
    all_errors.extend(errors_left.abs())
    all_errors.extend(errors_right.abs())

  return np.mean(all_errors) if all_errors else 0.0


def get_mean_errors(dataframes: list[pd.DataFrame],
                    ground_truth_col: str,
                    test_col: str) -> np.ndarray:
  """Calculates the mean errors for a given ground truth and test column.

  Args:
    dataframes: List of DataFrames containing calibration data.
    ground_truth_col: Column name for the ground truth values.
    test_col: Column name for the test values.

  Returns:
    np.ndarray: Array of mean errors for each frequency.
  """
  all_errors = []
  for df in dataframes:
    if ground_truth_col in df.columns and test_col in df.columns:
      errors = df[test_col] - df[ground_truth_col]
      all_errors.append(errors)
    else:
      print('Warning: Missing columns in DataFrame. Skipping this file.')
  all_errors = np.array(all_errors)
  return np.mean(all_errors, axis=0)


def plot_error_by_frequency(preprocessed_dfs: list[pd.DataFrame],
                            existing_cal_dict: dict[int, float],
                            new_cal: dict[int, float],
                            filtered_cal: dict[int, float] | None,
                            method_name: str,
                            left_col: str, right_col: str):
  """Produces a grouped bar chart of MAE at each frequency for 3 conditions."""
  if not preprocessed_dfs:
    return

  def get_mae_by_freq(dataframes):
    """Helper to calculate MAE for each frequency."""
    all_errors = []
    for df in dataframes:
      errors_left = df.copy()
      errors_left['error'] = (df[left_col] - df['Ground_truth_left']).abs()
      all_errors.append(errors_left[['Frequency_hz', 'error']])
      errors_right = df.copy()
      errors_right['error'] = (df[right_col] - df['Ground_truth_right']).abs()
      all_errors.append(errors_right[['Frequency_hz', 'error']])
    combined = pd.concat(all_errors)
    return combined.groupby('Frequency_hz')['error'].mean()

  # 1. "Before" condition
  mae_by_freq_before = get_mae_by_freq(preprocessed_dfs)
  mae_overall_before = _calculate_mae(preprocessed_dfs, left_col, right_col)

  # 2. "New Calibration" condition
  freqs = sorted(new_cal.keys())
  cal_adjustment_new = pd.Series(
      [new_cal[f] - existing_cal_dict.get(f, 0) for f in freqs],
      index=[str(f) for f in freqs]
  )
  corrected_dfs_new = []
  for df in preprocessed_dfs:
    corrected_df = df.copy()
    freq_str = corrected_df['Frequency_hz'].astype(str)
    adjustment_values = cal_adjustment_new[freq_str].values
    corrected_df[left_col] = corrected_df[left_col] - adjustment_values
    corrected_df[right_col] = corrected_df[right_col] - adjustment_values
    corrected_dfs_new.append(corrected_df)
  mae_by_freq_new = get_mae_by_freq(corrected_dfs_new)
  mae_overall_new = _calculate_mae(corrected_dfs_new, left_col, right_col)

  # --- Combine data for plotting ---
  plot_data = {
      f'Old calibration (MAE: {mae_overall_before:.2f} dB)': mae_by_freq_before,
      f'New calibration (MAE: {mae_overall_new:.2f} dB)': mae_by_freq_new,
  }

  # 3. "Filtered Calibration" condition (optional)
  if filtered_cal:
    cal_adjustment_filtered = pd.Series(
        [filtered_cal[f] - existing_cal_dict.get(f, 0) for f in freqs],
        index=[str(f) for f in freqs]
    )
    corrected_dfs_filtered = []
    for df in preprocessed_dfs:
      corrected_df = df.copy()
      freq_str = corrected_df['Frequency_hz'].astype(str)
      adjustment_values = cal_adjustment_filtered[freq_str].values
      corrected_df[left_col] = corrected_df[left_col] - adjustment_values
      corrected_df[right_col] = corrected_df[right_col] - adjustment_values
      corrected_dfs_filtered.append(corrected_df)
    mae_by_freq_filtered = get_mae_by_freq(corrected_dfs_filtered)
    mae_overall_filtered = _calculate_mae(
        corrected_dfs_filtered, left_col, right_col)
    plot_data[f'Filtered calibration (MAE: {mae_overall_filtered:.2f} dB)'] = \
        mae_by_freq_filtered

  mae_df = pd.DataFrame(plot_data)

  # --- Plotting ---
  _, ax = plt.subplots(figsize=(14, 7))
  colors = [COLOR_OLD_CAL, COLOR_NEW_CAL]
  if filtered_cal:
    colors.append(COLOR_FILTERED_CAL)
  mae_df.plot(kind='bar', ax=ax, width=0.8, color=colors)
  ax.set_title(
      f'Mean Absolute Error (MAE) by Frequency ({method_name})',
      fontsize=16)
  ax.set_xlabel('Frequency (Hz)', fontsize=12)
  ax.set_ylabel('Mean Absolute Error (dB)', fontsize=12)
  ax.tick_params(axis='both', which='major', labelsize=12, rotation=0)
  ax.grid(axis='y', linestyle='--', alpha=0.6)
  ax.legend(title='Calibration Type', fontsize=10)
  plt.tight_layout()
  plt.show()


def _apply_fir_filter(calibration_dict: dict[int, float],
                      a: float = 0.2) -> dict[int, float]:
  """Applies a simple FIR filter to the calibration data.

  Args:
    calibration_dict: Dict of calibration values, with frequencies as keys.
    a: The filter coefficient.

  Returns:
    A new dictionary with the filtered calibration values.
  """
  weights = [a, 1 - 2 * a, a]
  print(f'FIR filter applied with a={a}. '
        f'Weights: [{weights[0]:.2f}, {weights[1]:.2f}, {weights[2]:.2f}]')

  # Sort the dictionary by frequency to ensure correct order for filtering.
  sorted_freqs = sorted(calibration_dict.keys())
  cal_values = np.array([calibration_dict[f] for f in sorted_freqs])

  # Define the FIR filter weights.
  fir_filter = np.array([a, 1 - 2 * a, a])

  # Apply the filter with replicated edge conditions.
  filtered_values = scipy.ndimage.convolve1d(cal_values, fir_filter,
                                             mode='nearest')

  # Create a new dictionary with the filtered values.
  filtered_cal_dict = {freq: val for freq, val in zip(sorted_freqs,
                                                      filtered_values)}
  return filtered_cal_dict


def calculate_version2_calibration(
    dataframes: list[pd.DataFrame],
    filenames: list[str],
    existing_cal_dict: dict[int, float],
    method: str
) -> dict[int, float]:
  """Calculates calibration using the 'Version 2' method.

  Args:
    dataframes: A list of Pandas DataFrames, each from a calibration CSV file.
    filenames: A list of filenames corresponding to the dataframes.
    existing_cal_dict: A dictionary loaded from the existing_calibration file.
    method: The calibration method to use ('HW' or 'Adaptive').

  Returns:
    dict: A dictionary representing the new calculated calibration.
  """
  print('\n--- Starting Version 2 Calibration Calculation ---')
  if not dataframes:
    print('No calibration dataframes to process. Returning empty calibration.')
    return {}

  # --- Outlier Detection ---
  left_col, right_col = ('HW_left', 'HW_right') if method == 'HW' else \
                        ('Adaptive_mean_left', 'Adaptive_mean_right')
  subject_mean_errors = []
  for df in dataframes:
    errors_left = df[left_col] - df['Ground_truth_left']
    errors_right = df[right_col] - df['Ground_truth_right']
    abs_errors = pd.concat([errors_left.abs(), errors_right.abs()])
    subject_mean_errors.append(abs_errors.mean())

  subject_mean_errors = np.array(subject_mean_errors)
  mean_of_errors = np.mean(subject_mean_errors)
  std_of_errors = np.std(subject_mean_errors)
  outlier_threshold = mean_of_errors + OUTLIER_STD_THRESHOLD * std_of_errors

  outlier_indices = [i for i, error in enumerate(subject_mean_errors)
                     if error > outlier_threshold]

  if outlier_indices:
    print('\n--- Outlier Detection ---')
    # Sort for consistent removal and reporting.
    for i in sorted(outlier_indices, reverse=True):
      print(f'Excluding outlier: {filenames[i]} (Mean error '
            f'{subject_mean_errors[i]:.2f} dB > threshold '
            f'{outlier_threshold:.2f} dB)')
      # Remove from lists by index.
      dataframes.pop(i)
      filenames.pop(i)
  else:
    print('\n--- Outlier Detection ---')
    print('No outliers found.')
  # --- End of Outlier Detection ---

  print('\n--- Validating DataFrame Frequencies (Simplified) ---')
  all_frequencies_match = True
  sorted_dict_freqs = sorted([int(float(k)) for k in existing_cal_dict.keys()])

  # Check that all frequencies match the existing calibration dictionary.
  for df in dataframes:
    if 'Frequency_hz' not in df.columns:
      all_frequencies_match = False
      continue
    sorted_df_freqs = sorted([int(float(freq)) for freq in df['Frequency_hz']])
    if sorted_df_freqs != sorted_dict_freqs:
      all_frequencies_match = False
  if not all_frequencies_match:
    raise ValueError('One or more frequency set validation issues found.')

  method_name = 'Hughson-Westlake' if method == 'HW' else 'Adaptive'

  # Run the calibration logic based on the selected method.
  if method == 'HW':
    mean_errors_left = get_mean_errors(dataframes,
                                       ground_truth_col='Ground_truth_left',
                                       test_col='HW_left')
    mean_errors_right = get_mean_errors(dataframes,
                                        ground_truth_col='Ground_truth_right',
                                        test_col='HW_right')
  else:  # Adaptive PTA.
    mean_errors_left = get_mean_errors(dataframes,
                                       ground_truth_col='Ground_truth_left',
                                       test_col='Adaptive_mean_left')
    mean_errors_right = get_mean_errors(dataframes,
                                        ground_truth_col='Ground_truth_right',
                                        test_col='Adaptive_mean_right')
  # Create plots for audiograms and subject errors.
  plot_audiograms(dataframes, left_col, right_col, method_name)
  plot_subject_errors(dataframes, left_col, right_col, method_name)

  # New calibration calculation.
  calibration_adjustment = (mean_errors_left + mean_errors_right) / 2
  # Get the existing calibration values from the dictionary.
  existing_calibration_vector = []
  for freq in sorted_dict_freqs:
    existing_calibration_vector.append(existing_cal_dict[freq])
  # Calculate the new calibration.
  new_calibration_vector = []
  for i, freq in enumerate(sorted_dict_freqs):
    new_calibration_vector.append(existing_calibration_vector[i] +
                                   calibration_adjustment[i])
  # Create the new calibration dictionary.
  new_calibration_dict = {}
  for i, freq in enumerate(sorted_dict_freqs):
    new_calibration_dict[freq] = new_calibration_vector[i]
  return new_calibration_dict


def plot_audiograms(dataframes: list[pd.DataFrame],
                    left_col: str, right_col: str, method_name: str):
  """Plots ground truth and test audiograms."""
  _, ax = plt.subplots(figsize=(12, 6))
  num_subjects = len(dataframes)
  # Create color maps for ground truth (blues) and test (reds).
  colors_gt = plt.cm.Blues(np.linspace(0.3, 1, num_subjects))
  colors_test = plt.cm.Reds(np.linspace(0.3, 1, num_subjects))

  for i, df in enumerate(dataframes):
    freqs = df['Frequency_hz']
    # Plot ground truth audiograms with varying shades of blue.
    ax.plot(freqs, df['Ground_truth_left'], color=colors_gt[i], alpha=0.7)
    ax.plot(freqs, df['Ground_truth_right'], color=colors_gt[i], alpha=0.7)
    # Plot test audiograms with varying shades of red.
    ax.plot(freqs, df[left_col], color=colors_test[i], alpha=0.7)
    ax.plot(freqs, df[right_col], color=colors_test[i], alpha=0.7)

  # Add dummy lines for legend.
  ax.plot([], [], color='skyblue', label='Ground Truth')
  ax.plot([], [], color='salmon', label=f'{method_name} Test')
  ax.set_title(f'Ground Truth vs. Test Audiograms ({method_name})', fontsize=16)
  ax.set_ylabel('Hearing Level (dB HL)', fontsize=12)
  ax.set_ylim(85, -15)  # Invert y-axis for audiogram convention.
  ax.legend()
  ax.grid(True, linestyle='--', alpha=0.6)
  plt.tight_layout()
  plt.show()


def plot_subject_errors(dataframes: list[pd.DataFrame],
                      left_col: str, right_col: str, method_name: str):
  """Plots errors for each subject."""
  _, ax = plt.subplots(figsize=(12, 6))
  prop_cycle = plt.rcParams['axes.prop_cycle']
  colors = prop_cycle.by_key()['color']
  num_colors = len(colors)

  for i, df in enumerate(dataframes):
    color = colors[i % num_colors]  # Cycle through the default colors.
    errors_left = df[left_col] - df['Ground_truth_left']
    errors_right = df[right_col] - df['Ground_truth_right']
    ax.plot(df['Frequency_hz'], errors_left, color=color, alpha=0.7)
    ax.plot(df['Frequency_hz'], errors_right, color=color, alpha=0.7)

  ax.axhline(0, color='black', linestyle='--')
  ax.set_title(f'Error (Test - Ground Truth) per Subject ({method_name})',
               fontsize=16)
  ax.set_xlabel('Frequency (Hz)', fontsize=12)
  ax.set_ylabel('Error (dB)', fontsize=12)
  ax.grid(True, linestyle='--', alpha=0.6)
  plt.tight_layout()
  plt.show()


def main():
  parser = argparse.ArgumentParser(
    description='Analyze and plot calibration data from a CSV file/directory.')
  parser.add_argument('input_path',
                      help='Path to a calibration CSV file directory.')
  parser.add_argument(
      '--method',
      required=True,
      choices=['HW', 'Adaptive'],
      help='The test method: HW, for Hughson-Westlake, or Adaptive.')
  args = parser.parse_args()

  if os.path.isdir(args.input_path):
    process_calibration_directory(args.input_path, args.method)
  else:
    print(f'Error: Input path is not a valid directory: {args.input_path}')


if __name__ == '__main__':
  main()
