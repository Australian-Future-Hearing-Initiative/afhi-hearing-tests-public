"""Functions relating to the calculation or download of CLS test results."""

import io
import pandas as pd
import time

import streamlit as st

import common
from common import DEMO_UPDATED


def generate_cls_results_csv(df: pd.DataFrame,
                             duration_s: int) -> str:
  """Generates the CSV content string for the CLS results.

  Args:
    df: DataFrame containing the test results.
    duration_s: Duration of the test in seconds.

  Returns:
    String containing the formatted CSV data.
  """
  buffer = io.StringIO()
  buffer.write(f'# {DEMO_UPDATED}\n')
  buffer.write('# Test Method: adaptive\n')
  buffer.write('# Categorical Loudness Scaling Test Results\n')
  buffer.write(f"# Test date/time (UTC): {time.strftime('%Y-%m-%d %H:%M')}\n")
  buffer.write(f'# Number of stimuli: {len(df)}\n')
  buffer.write(f'# Test duration: {int(duration_s)} s\n')
  volume_str = common.get_macos_system_volume()
  buffer.write(f'# System volume: {volume_str}\n')
  buffer.write('#\n')
  # Round response times to 3 decimal places (nearest millisecond).
  df_copy = df.copy() # Avoid modifying original df.
  if 'Response Time (s)' in df_copy.columns:
    df_copy['Response Time (s)'] = df_copy['Response Time (s)'].round(3)
  df_copy.to_csv(buffer, index=True)
  return buffer.getvalue()

def display_interpretation():
  """Displays some text to help users interpret the test results."""
  st.write('\n\n')
  st.subheader('Interpreting the results')
  st.write('Your results can be used to tune a hearing assistance device, so '
           'that it is personalized to your hearing profile. This personalized '
           'model can also be used to predict how you will perceive sounds at '
           'different frequencies and amplitudes. The line in the graph '
           'above shows your inferred threshold of hearing for different '
           'frequencies. The lower the dB SPL value, the more sensitive your '
           'hearing is at that frequency.')
