"""Functions relating to the calculation or download of pip test results."""

import io
import time

import pandas as pd
import streamlit as st

import common
from common import DEMO_UPDATED


def generate_pip_full_results_csv(pip_results: list,
                                  pip_duration_s: int) -> str:
  """Generates the CSV content string for the full PIP results.

  Args:
    pip_results: List of tuples containing (ear, frequency, dbhl, heard,
      response_time_s).
    pip_duration_s: Duration of the test in seconds.

  Returns:
    String containing the formatted CSV data.
  """
  buffer = io.StringIO()
  buffer.write(f'# {DEMO_UPDATED}\n')
  buffer.write('# Pip-Based Pure-Tone Audiometry Test Results\n')
  buffer.write(f"# Test date/time (UTC): {time.strftime('%Y-%m-%d %H:%M')}\n")
  buffer.write(f'# Tones presented: {len(pip_results)}\n')
  buffer.write(f'# Test duration: {int(pip_duration_s)} s\n')
  volume_str = common.get_macos_system_volume()
  buffer.write(f'# System volume: {volume_str}\n')
  buffer.write('#\n')
  df = pd.DataFrame(pip_results,
                    columns=['ear', 'frequency', 'dbhl', 'heard',
                             'response_time_s'])
  # Round response times to 3 decimal places (nearest millisecond).
  if 'response_time_s' in df.columns:
    df['response_time_s'] = df['response_time_s'].round(3)
  df.to_csv(buffer, index=False)
  return buffer.getvalue()

def generate_pip_audiogram_csv(ear_label: str, audiogram_data: list) -> str:
  """Generates the CSV content string for a single ear PIP audiogram.

  Args:
    ear_label: 'Left' or 'Right' to use in the header.
    audiogram_data: List of (frequency, threshold) pairs.

  Returns:
    String containing the formatted CSV data.
  """
  buffer = io.StringIO()
  buffer.write(f'# {ear_label}-Ear Audiogram\n')
  buffer.write(f"# Test date/time (UTC): {time.strftime('%Y-%m-%d %H:%M')}\n")
  buffer.write('# Frequency (Hz), Threshold (dB HL)\n')
  for freq, thresh in audiogram_data:
    buffer.write(f'{freq},{thresh}\n')
  return buffer.getvalue()

def display_interpretation():
  """Displays some text to help users interpret the test results."""
  st.write('\n\n')
  st.subheader('Interpreting the results')
  st.write('The audiogram above displays your hearing thresholds, which are '
           'the quietest sounds that could be heard, for each frequency '
           'tested. Lower dB thresholds (higher points on the y-axis scale) '
           'indicate better hearing.')
