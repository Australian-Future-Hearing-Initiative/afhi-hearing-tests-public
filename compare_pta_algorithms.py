"""Comparison tool for Pure-Tone Audiometry (PTA) algorithms.

This script analyzes and visualizes the performance of different PTA algorithms
by comparing their output against ground truth audiograms. It processes
simulation data to calculate key metrics including:
- Mean Absolute Difference (MAD)
- Test Duration
- Frequency-specific error
- 4-frequency and 8-frequency PTA errors

Example:
  python compare_pta_algorithms.py pta_comparison_data/synthetic_examples
"""

import argparse
import glob
import os
import sys
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

AUDIOGRAM_SUFFIX = '_audiograms.csv'
DURATION_SUFFIX = '_test_duration.csv'
SIGNIFICANCE_THRESHOLD = 0.05
PLOT_COLORS = ['blue', 'orange', 'green', 'red', 'purple']


def _calculate_pta_error(df_audio, method, freqs, error_key):
  """Calculate PTA error for given frequencies."""
  mask = df_audio['frequency_hz'].isin(freqs)
  if not mask.any():
    return []

  gt_left = df_audio.loc[mask, 'ground_truth_left'].mean()
  gt_right = df_audio.loc[mask, 'ground_truth_right'].mean()
  m_left = df_audio.loc[mask, f'{method}_left'].mean()
  m_right = df_audio.loc[mask, f'{method}_right'].mean()

  return [
    {'Ear': 'Left', 'Method': method, error_key: abs(m_left - gt_left)},
    {'Ear': 'Right', 'Method': method, error_key: abs(m_right - gt_right)},
  ]


def load_and_validate_data(data_dir: str) -> tuple[
  pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
  """Loads and validates audiogram and duration data from CSV files.

  Args:
    data_dir: Directory containing the CSV files.

  Returns:
    A tuple containing:
      - error_df: DataFrame with MAD errors for each subject and method.
      - duration_df: DataFrame with durations for each subject and method.
      - freq_diff_df: DataFrame with signed differences per frequency.
      - pta4_error_df: DataFrame with 4-PTA absolute errors.
      - pta8_error_df: DataFrame with 8-PTA absolute errors.
  """
  audiogram_files = glob.glob(os.path.join(data_dir, f'*{AUDIOGRAM_SUFFIX}'))

  if not audiogram_files:
    print(f'No files found with suffix {AUDIOGRAM_SUFFIX} in {data_dir}')
    sys.exit(1)

  mad_records = []
  duration_records = []
  freq_diffs = []
  pta4_errors = []
  pta8_errors = []

  print(f'Found {len(audiogram_files)} subjects to process.')

  for audio_file in audiogram_files:
    base_name = os.path.basename(audio_file).replace(AUDIOGRAM_SUFFIX, '')
    duration_file = os.path.join(data_dir, base_name + DURATION_SUFFIX)

    if not os.path.exists(duration_file):
      print(f'Error: Missing duration file for {base_name}')
      sys.exit(1)

    df_audio = pd.read_csv(audio_file)
    df_duration = pd.read_csv(duration_file)

    required_audio_cols = [
      'frequency_hz', 'ground_truth_left', 'ground_truth_right'
    ]
    if not all(col in df_audio.columns for col in required_audio_cols):
      print(f'Error: {audio_file} missing cols: {required_audio_cols}')
      sys.exit(1)

    method_cols_left = [
      c for c in df_audio.columns
      if c.endswith('_left') and c != 'ground_truth_left'
    ]
    methods = [c.replace('_left', '') for c in method_cols_left]

    if not methods:
      print(f'Error: No method columns found in {audio_file}')
      sys.exit(1)

    required_dur_cols = ['type', 'duration_s']
    if not all(col in df_duration.columns for col in required_dur_cols):
      print(f'Error: {duration_file} missing cols: {required_dur_cols}')
      sys.exit(1)

    duration_methods = set(df_duration['type'].unique())
    for method in methods:
      if f'{method}_right' not in df_audio.columns:
        print(f'Error: {audio_file} missing right ear for {method}')
        sys.exit(1)
      if method not in duration_methods:
        print(f'Error: {duration_file} missing duration for {method}')
        sys.exit(1)

    for method in methods:
      diffs_left = df_audio[f'{method}_left'] - df_audio['ground_truth_left']
      diffs_right = (
        df_audio[f'{method}_right'] - df_audio['ground_truth_right']
      )

      for i, freq in enumerate(df_audio['frequency_hz']):
        freq_diffs.extend([
          {'Subject': base_name, 'Method': method,
           'Frequency': freq, 'Difference': diffs_left.iloc[i]},
          {'Subject': base_name, 'Method': method,
           'Frequency': freq, 'Difference': diffs_right.iloc[i]},
        ])

      combined_abs_diffs = pd.concat([diffs_left.abs(), diffs_right.abs()])
      mad_records.append({
        'Subject': base_name, 'Method': method,
        'MAD': combined_abs_diffs.mean()
      })

      dur_val = df_duration.loc[
        df_duration['type'] == method, 'duration_s'
      ].values[0]
      duration_records.append({
        'Subject': base_name, 'Method': method, 'Duration (s)': dur_val
      })

      for rec in _calculate_pta_error(
        df_audio, method, [500, 1000, 2000, 4000], '4PTA_Error'
      ):
        rec['Subject'] = base_name
        pta4_errors.append(rec)

      for rec in _calculate_pta_error(
        df_audio, method,
        [250, 500, 1000, 2000, 3000, 4000, 6000, 8000], '8PTA_Error'
      ):
        rec['Subject'] = base_name
        pta8_errors.append(rec)

  return (
    pd.DataFrame(mad_records),
    pd.DataFrame(duration_records),
    pd.DataFrame(freq_diffs),
    pd.DataFrame(pta4_errors),
    pd.DataFrame(pta8_errors),
  )


def add_significance_brackets(ax, data_df, metric_col, methods):
  """Calculates stats and draws significance brackets on the plot."""
  pairs = list(combinations(methods, 2))
  significant_pairs = []

  for m1, m2 in pairs:
    data1 = data_df[data_df['Method'] == m1].sort_values('Subject')[
      metric_col].values
    data2 = data_df[data_df['Method'] == m2].sort_values('Subject')[
      metric_col].values
    _, p = stats.ttest_rel(data1, data2)
    if p < SIGNIFICANCE_THRESHOLD:
      significant_pairs.append((m1, m2, p))

  if not significant_pairs:
    return False

  y_max = data_df[metric_col].max()
  y_range = y_max - data_df[metric_col].min()
  bracket_h = y_range * 0.05
  text_offset = y_range * 0.02
  x_map = {m: i + 1 for i, m in enumerate(methods)}
  current_y = y_max + y_range * 0.1
  step_y = y_range * 0.1

  for m1, m2, p in significant_pairs:
    x1, x2 = x_map[m1], x_map[m2]
    y_top = current_y + bracket_h
    ax.plot([x1, x1, x2, x2], [current_y, y_top, y_top, current_y],
            color='black', lw=1.5)

    sig_text = 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'
    ax.text((x1 + x2) / 2, y_top + text_offset, sig_text,
            ha='center', va='bottom', fontsize=9)
    current_y += step_y

  ax.set_ylim(top=current_y + step_y)
  return True


def plot_metric_comparison(
  df: pd.DataFrame,
  metric_col: str,
  title: str,
  ylabel: str,
  method_order: list = None,
):
  """Generates box and whisker plots for a given metric."""
  fig, ax = plt.subplots(figsize=(10, 7))
  methods = method_order if method_order is not None else df['Method'].unique()

  data = [df[df['Method'] == m][metric_col].values for m in methods]
  bplot = ax.boxplot(
    data, tick_labels=methods, patch_artist=True,
    medianprops={'color': 'black'}, showmeans=True,
    meanprops={'marker': 'D', 'markeredgecolor': 'black',
               'markerfacecolor': 'white'},
  )
  legend_handles = []
  legend_labels = []
  for patch, color, method, method_data in zip(
    bplot['boxes'], PLOT_COLORS, methods, data
  ):
    patch.set_facecolor(color)
    mean_val = np.mean(method_data)
    legend_handles.append(patch)
    legend_labels.append(f'{method} (Mean: {mean_val:.1f})')

  ax.legend(legend_handles, legend_labels)

  ax.set_title(title)
  ax.set_ylabel(ylabel)
  ax.grid(True, axis='y')

  if df[metric_col].min() >= 0:
    ax.set_ylim(bottom=0)

  sig_found = add_significance_brackets(ax, df, metric_col, methods)
  legend_text = (
    f'Horizontal bars indicate significance at p < {SIGNIFICANCE_THRESHOLD}'
    if sig_found else
    f'No significant differences found (p > {SIGNIFICANCE_THRESHOLD})'
  )

  fig.text(0.5, 0.02, legend_text, ha='center', fontsize=10, color='black',
           bbox={'facecolor': 'white', 'edgecolor': 'lightgray', 'alpha': 0.8})

  plt.tight_layout(rect=[0, 0.05, 1, 1])
  plt.show()


def plot_frequency_error(freq_diff_df: pd.DataFrame):
  """Generates a bar plot of mean signed difference per frequency."""
  _, ax = plt.subplots(figsize=(12, 6))
  methods = freq_diff_df['Method'].unique()

  freq_medians = (
    freq_diff_df.groupby(['Method', 'Frequency'])['Difference']
    .median().reset_index()
  )
  frequencies = sorted(freq_medians['Frequency'].unique())

  x = np.arange(len(frequencies))
  width = 0.8 / len(methods)

  for i, method in enumerate(methods):
    method_data = freq_medians[freq_medians['Method'] == method]
    medians = [
      method_data[method_data['Frequency'] == f]['Difference'].values[0]
      if f in method_data['Frequency'].values else 0
      for f in frequencies
    ]
    offset = width * i - (width * (len(methods) - 1) / 2)
    ax.bar(x + offset, medians, width, label=method, color=PLOT_COLORS[i])

  ax.set_title('Median Signed Difference per Frequency')
  ax.set_ylabel('Median Difference (dB HL)')
  ax.set_xlabel('Frequency (Hz)')
  ax.set_xticks(x)
  ax.set_xticklabels(frequencies)
  ax.legend()
  ax.grid(True, axis='y')
  ax.axhline(0, color='black', linewidth=0.8)

  plt.tight_layout()
  plt.show()


def main():
  parser = argparse.ArgumentParser(
    description='Compare PTA algorithms against ground truth.'
  )
  parser.add_argument(
    'data_dir', type=str,
    help='Directory containing the audiogram and duration CSV files.',
  )
  args = parser.parse_args()

  print('Starting PTA Algorithm Comparison...')
  mad_df, duration_df, freq_diff_df, pta4_error_df, pta8_error_df = (
    load_and_validate_data(args.data_dir)
  )

  print('\n--- Summary Statistics (MAD) ---')
  print(mad_df.groupby('Method')['MAD'].describe())

  print('\n--- Summary Statistics (Duration) ---')
  print(duration_df.groupby('Method')['Duration (s)'].describe())

  if not pta4_error_df.empty:
    print('\n--- Summary Statistics (4-PTA Error) ---')
    print(pta4_error_df.groupby('Method')['4PTA_Error'].describe())

  if not pta8_error_df.empty:
    print('\n--- Summary Statistics (8-PTA Error) ---')
    print(pta8_error_df.groupby('Method')['8PTA_Error'].describe())

  plot_metric_comparison(
    mad_df, 'MAD',
    'Mean absolute difference (MAD) to ground truth across all frequencies',
    'dBHL',
  )

  plot_metric_comparison(
    duration_df, 'Duration (s)', 'Test Duration', 'Time (seconds)',
  )

  plot_frequency_error(freq_diff_df)

  method_order = mad_df['Method'].unique()

  if not pta4_error_df.empty:
    plot_metric_comparison(
      pta4_error_df, '4PTA_Error',
      '4-PTA absolute error from ground truth',
      'dBHL', method_order=method_order,
    )

  if not pta8_error_df.empty:
    plot_metric_comparison(
      pta8_error_df, '8PTA_Error',
      '8-PTA absolute error from ground truth',
      'dBHL', method_order=method_order,
    )


if __name__ == '__main__':
  main()
