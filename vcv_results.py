"""Functions for displaying and exporting adaptive VCV test results."""

import glob
import io
import os
import shutil
import time
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
import streamlit as st
import numpy as np
import pandas as pd

import bayesian_vcv_estimator
import common


def _collect_wav_files() -> list[tuple[str, bytes]]:
  """Reads saved WAV files from the temp directory as bytes.

  Returns:
    A list of (filename_in_zip, wav_bytes) tuples, or an empty list
    if no WAV directory is configured or no files are found.
  """
  wav_dir = st.session_state.get('vcv_wav_save_dir')
  if not wav_dir or not os.path.isdir(wav_dir):
    return []
  wav_paths = sorted(glob.glob(os.path.join(wav_dir, '*.wav')))
  wav_entries = []
  for wav_path in wav_paths:
    with open(wav_path, 'rb') as f:
      wav_entries.append(
          (f'audio/{os.path.basename(wav_path)}', f.read()))
  return wav_entries


def _cleanup_wav_dir():
  """Removes the temporary WAV directory after files have been zipped."""
  wav_dir = st.session_state.get('vcv_wav_save_dir')
  if wav_dir and os.path.isdir(wav_dir):
    shutil.rmtree(wav_dir)
    print(f'Removed temporary WAV directory: {wav_dir}')
    st.session_state.vcv_wav_save_dir = None


def create_confusion_matrix_image(confusion_matrix, labels):
  """Creates and returns a confusion matrix figure.
  """
  sns.set(font_scale=0.6)
  fig, ax = plt.subplots(figsize=(5, 4))
  sns.heatmap(confusion_matrix, annot=False, cmap='viridis', ax=ax,
              xticklabels=labels, yticklabels=labels, square=True)
  ax.set_xlabel('Responses')
  ax.set_ylabel('Target Stimuli')
  fig.tight_layout()
  return fig

def create_srt_plot(
    estimates_df: pd.DataFrame,
    ordered_labels: list,
):
  """Creates a floating-bar plot of SRTs by consonant and ear.

  Each bar spans the ±1 SD uncertainty interval around the
  SRT estimate, with a tick mark at the point estimate.
  This avoids an arbitrary zero baseline.
  """
  if estimates_df.empty:
    return None

  fig, ax = plt.subplots(figsize=(10, 6))
  ears_present = estimates_df['Ear'].unique()
  hue_order = (
      ['both'] if 'both' in ears_present
      else sorted(
          e for e in ['left', 'right']
          if e in ears_present
      )
  )
  # Match the PTA demo's left/right ear colours.
  palette = {
      'left': '#87CEEB',
      'right': '#FA8072',
      'both': '#B0C4DE',
  }

  present_consonants = estimates_df['Consonant'].unique()
  consonant_order = [
      c for c in ordered_labels if c in present_consonants
  ]

  n_consonants = len(consonant_order)
  n_hues = len(hue_order)
  group_width = 0.8
  bar_width = group_width / n_hues
  x_positions = np.arange(n_consonants)

  for j, ear in enumerate(hue_order):
    offset = (j - (n_hues - 1) / 2.0) * bar_width
    color = palette[ear]

    for i, consonant in enumerate(consonant_order):
      row = estimates_df[
          (estimates_df['Consonant'] == consonant)
          & (estimates_df['Ear'] == ear)
      ]
      if row.empty:
        continue

      srt = row['SRT (dB)'].iloc[0]
      sd = row['Uncertainty (SD)'].iloc[0]
      if np.isnan(srt) or np.isnan(sd):
        continue

      x = x_positions[i] + offset
      bottom = srt - sd
      height = 2.0 * sd

      # Floating bar spanning [SRT-SD, SRT+SD].
      ax.bar(
          x, height, width=bar_width,
          bottom=bottom, color=color,
          edgecolor='white', linewidth=0.5,
          alpha=0.85,
      )
      # Tick mark at the point estimate.
      half_w = bar_width * 0.45
      ax.plot(
          [x - half_w, x + half_w],
          [srt, srt],
          color='black', linewidth=1.5,
      )

  # Build the legend manually.
  handles = [
      Patch(
          facecolor=palette[e], alpha=0.85,
          label=f'Likely SRT range ({e})',
      )
      for e in hue_order
  ]
  handles.append(
      Line2D(
          [0], [0], color='black', linewidth=1.5,
          label='SRT estimate',
      )
  )
  ax.legend(handles=handles, fontsize=11)

  # Draw class group separators and labels.
  classes = bayesian_vcv_estimator.CONSONANT_CLASSES
  display_order = (
      bayesian_vcv_estimator.CLASS_DISPLAY_ORDER
  )
  idx = 0
  for cls_name in display_order:
    members = classes[cls_name]['members']
    # Only count members that are actually in the plot.
    n = sum(1 for m in members if m in consonant_order)
    if n == 0:
      continue
    # Vertical dashed separator before each group
    # (skip the very first one).
    if idx > 0:
      ax.axvline(
          idx - 0.5, color='grey',
          linestyle=':', linewidth=1,
      )
    # Class label inside the plot, near the top.
    centre_x = idx + (n - 1) / 2.0
    ax.text(
        centre_x, 0.96, cls_name,
        transform=ax.get_xaxis_transform(),
        ha='center', va='top',
        fontsize=10, fontstyle='italic',
        color='#444444',
    )
    idx += n

  ax.set_xticks(x_positions)
  ax.set_xticklabels(consonant_order)
  ax.set_title(
      'Speech Reception Thresholds (SRT) '
      'by Consonant and Ear',
      fontsize=14,
  )
  ax.set_ylabel('SRT (dB SNR)', fontsize=12)
  ax.set_xlabel('Consonant', fontsize=12)

  # Extend y-axis by 5 dB in each direction for clarity.
  y_lo, y_hi = ax.get_ylim()
  ax.set_ylim(y_lo - 5, y_hi + 5)

  ax.grid(axis='y', linestyle='--', alpha=0.7)
  fig.tight_layout()
  return fig

def display_results(
    results_left, results_right, all_possible_labels, df,
    merge_lr, n_tests
):
  """Displays the results for both ears.

  Args:
      results_left: Analysis results dict for the left ear (or None).
      results_right: Analysis results dict for the right ear (or None).
      all_possible_labels: List of all possible consonant labels.
      df: DataFrame containing combined test results.
      merge_lr: Boolean indicating if L/R were merged.
      n_tests: Number of tests per ear.
  """
  st.write('\n\n')
  st.subheader('Test completed. Thank you for participating!')

  col1, col2 = st.columns(2)
  fig_left, fig_right = None, None

  with col1:
    st.write('#### Left Ear')
    if results_left:
      accuracy_left = results_left['accuracy'] * 100
      st.write(f'Accuracy: {accuracy_left:.1f}%')
      st.write('Confusion Matrix:')
      fig_left = create_confusion_matrix_image(
        results_left['confusion_matrix'],
        all_possible_labels
      )
      st.pyplot(fig_left)
    else:
      st.write('No results available for the left ear.')

  with col2:
    st.write('#### Right Ear')
    if results_right:
      accuracy_right = results_right['accuracy'] * 100
      st.write(f'Accuracy: {accuracy_right:.1f}%')
      st.write('Confusion Matrix:')
      fig_right = create_confusion_matrix_image(
        results_right['confusion_matrix'],
        all_possible_labels
      )
      st.pyplot(fig_right)
    else:
      st.write('No results available for the right ear.')

  # Generate CSV content/
  buffer = io.StringIO()
  buffer.write(f'# {common.DEMO_UPDATED}\n')
  buffer.write('# Consonant Confusion Test Results\n')
  buffer.write(f"# Test date/time (UTC): {time.strftime('%Y-%m-%d %H:%M')}\n")
  if merge_lr:
    buffer.write('# Test Mode: Merged L/R\n')
  else:
    buffer.write('# Test Mode: Separate L/R\n')
  buffer.write(f'# Number of stimuli per ear: {n_tests}\n')
  volume_str = common.get_macos_system_volume()
  buffer.write(f'# System volume: {volume_str}\n')
  buffer.write('#\n')
  df_to_save = df.copy()
  if 'Response Time (s)' in df_to_save.columns:
    df_to_save['Response Time (s)'] = df_to_save['Response Time (s)'].round(3)
  df_to_save.to_csv(buffer, index=True)
  full_results_csv_content = buffer.getvalue()

  # Prepare list of files for zip.
  files_for_zip = [
      ('vcv_full_results.csv', full_results_csv_content)
  ]
  if fig_left:
    files_for_zip.append(('vcv_confusion_matrix_left.png', fig_left))
  if fig_right:
    files_for_zip.append(('vcv_confusion_matrix_right.png', fig_right))
  # Include saved WAV audio files if available (NAL + local only).
  files_for_zip.extend(_collect_wav_files())

  zip_prefix = 'vcv_results'
  test_name = 'Consonant Confusion Test'
  # Display download button.
  zip_data = common.generate_zip_bytes(files_for_zip)
  timestamp = datetime.now().strftime('%Y%m%d_%H%M')
  zip_filename = f'UTC{timestamp}_{zip_prefix}.zip'

  # Save local backup if applicable; set flag to prevent multiple backups.
  if (st.session_state.is_running_locally and
      st.session_state.app_target_audience == 'NAL' and
      not st.session_state.get('vcv_backup_saved', False)):
    common.save_local_backup(zip_data, zip_filename)
    st.session_state.vcv_backup_saved = True
    print('VCV local backup saved.')

  # Clean up the temporary WAV directory now that files are in the zip.
  _cleanup_wav_dir()

  st.write('\n\n')  # Add some space before the download button.
  st.download_button(
      label='Download results',
      data=zip_data,
      file_name=zip_filename,
      mime='application/zip',
      key='vcv_manual_download'
  )
  # Display email form.
  common.display_email_results_form(test_name, files_for_zip, zip_prefix)

  # Close figures if they were created.
  if fig_left:
    plt.close(fig_left)
  if fig_right:
    plt.close(fig_right)

def display_adaptive_results(
    estimates_df: pd.DataFrame,
    raw_data_df: pd.DataFrame,
    total_trials: int,
    confusion_results: dict,
    all_possible_labels: list
):
  """Displays the results for the adaptive VCV test."""
  st.write('\n\n')
  st.subheader('Test completed. Thank you for participating!')
  st.write('The results show the estimated Speech Reception Threshold (SRT) in '
           'dB SNR for each consonant. A lower (more negative) SRT indicates '
           'better performance.')

  if estimates_df is None or estimates_df.empty:
    st.warning('No results available.')
    return

  display_df = estimates_df.copy()
  display_df['SRT (dB)'] = display_df['SRT (dB)'].round(2)
  display_df['Uncertainty (SD)'] = display_df['Uncertainty (SD)'].round(2)
  is_merged = 'both' in estimates_df['Ear'].unique()

  st.write('#### SRT Visualization')
  fig_srt = create_srt_plot(estimates_df, all_possible_labels)
  if fig_srt:
    st.pyplot(fig_srt)

  st.write('#### SRT Results Summary')
  display_df['Consonant'] = pd.Categorical(
      display_df['Consonant'], categories=all_possible_labels, ordered=True
  )
  display_df = display_df.sort_values('Consonant')

  if is_merged:
    st.write('##### Binaural (Both Ears)')
    st.dataframe(
        display_df[display_df['Ear'] == 'both']
        .drop(columns=['Ear'])
        .set_index('Consonant')
    )
  else:
    col1, col2 = st.columns(2)
    with col1:
      st.write('##### Left Ear')
      left_df = display_df[display_df['Ear'] == 'left'].drop(columns=['Ear'])
      if not left_df.empty:
        st.dataframe(left_df.set_index('Consonant'))
      else:
        st.write('No left ear results.')
    with col2:
      st.write('##### Right Ear')
      right_df = display_df[display_df['Ear'] == 'right'].drop(columns=['Ear'])
      if not right_df.empty:
        st.dataframe(right_df.set_index('Consonant'))
      else:
        st.write('No right ear results.')

  st.write('#### Confusion Matrices (Overall)')

  col1, col2 = st.columns(2)
  fig_left, fig_right = None, None

  # Helper to display matrix
  def show_matrix(col, ear_label, key):
    with col:
      st.write(f'##### {ear_label}')
      results = confusion_results.get(key)
      if results:
        fig = create_confusion_matrix_image(
            results['confusion_matrix'], all_possible_labels
        )
        st.pyplot(fig)
        return fig
      st.write(f'No {ear_label.lower()} data.')
      return None

  if is_merged:
    fig_left = show_matrix(col1, 'Binaural (Both Ears)', 'both')
  else:
    fig_left = show_matrix(col1, 'Left Ear', 'left')
    fig_right = show_matrix(col2, 'Right Ear', 'right')

  # Generate CSV content/
  buffer = io.StringIO()
  buffer.write(f'# {common.DEMO_UPDATED}\n')
  buffer.write('# Adaptive Consonant Test Results\n')
  buffer.write(f"# Test date/time (UTC): {time.strftime('%Y-%m-%d %H:%M')}\n")
  mode_str = 'Binaural (Merged)' if is_merged else 'Monaural (Interleaved)'
  buffer.write(f'# Test Mode: {mode_str}\n')
  buffer.write(f'# Total trials: {total_trials}\n')
  volume_str = common.get_macos_system_volume()
  buffer.write(f'# System volume: {volume_str}\n')
  buffer.write('#\n')

  df_to_save_raw = raw_data_df.copy()
  if 'Response Time (s)' in df_to_save_raw.columns:
    df_to_save_raw['Response Time (s)'] = df_to_save_raw[
        'Response Time (s)'
    ].round(3)
  df_to_save_raw.to_csv(buffer, index=True)
  raw_results_csv_content = buffer.getvalue()

  buffer_est = io.StringIO()
  estimates_df.to_csv(buffer_est, index=False, float_format='%.3f')
  estimates_csv_content = buffer_est.getvalue()

  # Prepare list of files for zip.
  files_for_zip = [
      ('vcv_raw_trial_data.csv', raw_results_csv_content),
      ('vcv_srt_estimates.csv', estimates_csv_content),
  ]
  if fig_srt:
    files_for_zip.append(('vcv_srt_visualization.png', fig_srt))
  if fig_left:
    fname = (
        'vcv_confusion_matrix_both.png'
        if is_merged
        else 'vcv_confusion_matrix_left.png'
    )
    files_for_zip.append((fname, fig_left))
  if fig_right:
    files_for_zip.append(('vcv_confusion_matrix_right.png', fig_right))
  # Include saved WAV audio files if available (NAL + local only).
  files_for_zip.extend(_collect_wav_files())

  zip_prefix = 'vcv_adaptive_results'
  test_name = 'Adaptive Consonant Test'

  # Display download button.
  zip_data = common.generate_zip_bytes(files_for_zip)
  timestamp = datetime.now().strftime('%Y%m%d_%H%M')
  zip_filename = f'UTC{timestamp}_{zip_prefix}.zip'

  # Save local backup if applicable; set flag to prevent multiple backups.
  if (st.session_state.is_running_locally and
      st.session_state.app_target_audience == 'NAL' and
      not st.session_state.get('vcv_backup_saved', False)):
    common.save_local_backup(zip_data, zip_filename)
    st.session_state.vcv_backup_saved = True
    print('VCV local backup saved.')

  # Clean up the temporary WAV directory now that files are in the zip.
  _cleanup_wav_dir()

  st.write('\n\n')  # Add some space before the download button.
  st.download_button(
      label='Download results',
      data=zip_data,
      file_name=zip_filename,
      mime='application/zip',
      key='vcv_manual_download'
  )
  # Display email form.
  common.display_email_results_form(test_name, files_for_zip, zip_prefix)

  # Close figures if they were created.
  if fig_left:
    plt.close(fig_left)
  if fig_right:
    plt.close(fig_right)
  if fig_srt:
    plt.close(fig_srt)

def display_constant_interpretation():
  """Displays an example to help users interpret the test results."""
  st.write('\n\n')
  st.subheader('Interpreting the results')
  st.write('The confusion matrix shows the number of times each '
           'consonant was confused with another. This information can help '
           'identify which sounds are most difficult to distinguish for you. '
           'As an example, the confusion matrices below show results for '
           'the same subject, without (left) and with (earplugs), to '
           'simulate high-frequency hearing loss.')
  st.image('assets/cm_examples.png', width=600)
  st.subheader('Further reading')
  st.markdown('This demo is based on the work of '
              '[Hajicek, Harris, Neely. J. Acoust. Soc. Am. '
              '154, A34 (2023)](https://doi.org/10.1121/10.0022707)')

def display_adaptive_interpretation():
  """Displays an example to help users interpret the test results."""
  st.write('\n\n')
  st.subheader('Interpreting the results')
  st.write('The Speech Reception Threshold (SRT) is the estimated '
           'Signal-to-Noise Ratio (SNR) required for you to correctly identify '
           'a specific consonant roughly 55% of the time (accounting for '
           'guessing).')
  st.write('**Lower SRT values (more negative in dB) indicate better hearing '
           'performance.** This means you can still distinguish that consonant '
           'in more noise.')
  st.write('The error bars around each SRT estimate indicate the confidence '
           'in that estimate.')
  st.subheader('Further reading')
  st.markdown('This demo is based on the work of '
              '[Hajicek, Harris, Neely. J. Acoust. Soc. Am. '
              '154, A34 (2023)](https://doi.org/10.1121/10.0022707)')
  st.markdown('This adaptive procedure is based on the ZEST algorithm: '
              '[King-Smith, P. E., et al. (1994). Efficient and unbiased '
              'modifications of the QUEST threshold-estimation method. '
              'Vision research, 34(7), 885-912.]'
              '(https://doi.org/10.1016/0042-6989(94)90039-6)')
