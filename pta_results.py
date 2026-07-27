"""Functions relating to the calculation or download of PTA test results."""

import io
import time

import pandas as pd
import streamlit as st

import common
from common import DEMO_UPDATED


def generate_pta_full_results_csv(pta_results: list,
                                  pta_duration_s: int,
                                  pta_method: str,
                                  active_duration_s: float = None,
                                  pause_count: int = 0,
                                  total_pause_duration_s: float = 0.0) -> str:
  """Generates the CSV content string for the full PTA results.

  Args:
    pta_results: List of tuples containing (ear, frequency, dbhl, heard,
      response_time_s).
    pta_duration_s: Duration of the test in seconds.
    pta_method: The name of the PTA method used.
    active_duration_s: Duration of active testing in seconds (excluding pause).
    pause_count: Total number of pause events during test.
    total_pause_duration_s: Total time spent in paused state in seconds.

  Returns:
    String containing the formatted CSV data.
  """
  buffer = io.StringIO()
  buffer.write(f'# {DEMO_UPDATED}\n')
  buffer.write(f'# Test Method: {pta_method}\n')
  buffer.write('# Pure-Tone Audiometry Test Results\n')
  buffer.write(f"# Test date/time (UTC): {time.strftime('%Y-%m-%d %H:%M')}\n")
  buffer.write(f'# Tones presented: {len(pta_results)}\n')
  buffer.write(f'# Test duration: {int(pta_duration_s)} s\n')
  if active_duration_s is not None:
    buffer.write(f'# Active duration: {int(active_duration_s)} s\n')
    buffer.write(f'# Pause count: {pause_count}\n')
    buffer.write(f'# Total paused duration: {int(total_pause_duration_s)} s\n')
  volume_str = common.get_macos_system_volume()
  buffer.write(f'# System volume: {volume_str}\n')
  buffer.write('#\n')
  df = pd.DataFrame(pta_results,
                    columns=['ear', 'frequency', 'dbhl', 'heard',
                             'response_time_s'])
  df['response_time_s'] = df['response_time_s'].round(3)
  df.to_csv(buffer, index=False)
  return buffer.getvalue()

def generate_audiogram_csv(ear_label: str, audiogram_data: list) -> str:
  """Generates the CSV content string for a single ear audiogram.

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

