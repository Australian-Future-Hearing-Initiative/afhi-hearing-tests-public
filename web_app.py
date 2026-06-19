"""The main web application for the AFHI demos deployed on Streamlit Cloud.

The idea behind this module is to provide a user-friendly interface for the
various demos. The user can select a demo from the sidebar. This then runs the
corresponding function to display the demo. The goal is to make this highly
extensible, so that new demos can be added easily.
"""

import streamlit as st
import os

import demo_cls
import demo_vcv
import demo_pip
import demo_pta
import demo_tone_generator
import common


CSS_PATH = 'styles.css'  # Button styling for all demos is in this CSS file.

DEMOS = [
    'Pure-Tone Audiometry',
    'Pip PTA',
    'Consonant Confusion Test',
    'Categorical Loudness Scaling',
    'Tone Generator & Calibration',
]

# Map demo names to their corresponding initialization functions.
DEMO_FUNCTIONS = {
    'Pure-Tone Audiometry': demo_pta.create_main_demo,
    'Categorical Loudness Scaling': demo_cls.create_main_demo,
    'Consonant Confusion Test': demo_vcv.create_main_demo,
    'Pip PTA': demo_pip.create_main_demo,
    'Tone Generator & Calibration': demo_tone_generator.create_main_demo,
}

def load_css(css_file_path):
  """Loads a CSS file to style all demos in the AFHI web app."""
  with open(css_file_path, encoding='utf-8') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def reset_session_state():
  """Clears all session state and restores essential app variables."""
  # Store the values we need to keep.
  current_demo = st.session_state.current_demo
  audio_container = st.session_state.audio_container
  app_target_audience = st.session_state.app_target_audience
  is_running_locally = st.session_state.is_running_locally

  # Preserve calibration settings across demo resets
  dynamic_offsets = st.session_state.get('dynamic_offsets')
  dynamic_calibrated_device = st.session_state.get('dynamic_calibrated_device')
  dynamic_cal_res = st.session_state.get('dynamic_cal_res')
  pta_custom_device = st.session_state.get('pta_custom_device')
  pip_custom_device = st.session_state.get('pip_custom_device')

  # Clear everything.
  st.session_state.clear()
  # Restore essential variables.
  st.session_state.current_demo = current_demo
  st.session_state.audio_container = audio_container
  st.session_state.app_target_audience = app_target_audience
  st.session_state.is_running_locally = is_running_locally

  # Restore calibration variables
  if dynamic_offsets is not None:
    st.session_state.dynamic_offsets = dynamic_offsets
  if dynamic_calibrated_device is not None:
    st.session_state.dynamic_calibrated_device = dynamic_calibrated_device
  if dynamic_cal_res is not None:
    st.session_state.dynamic_cal_res = dynamic_cal_res
  if pta_custom_device is not None:
    st.session_state.pta_custom_device = pta_custom_device
  if pip_custom_device is not None:
    st.session_state.pip_custom_device = pip_custom_device

  st.session_state.initial_state_set = True


def create_sidebar():
  """Creates the sidebar with instructions and demo selection."""
  with st.sidebar:
    st.header('Australian Future Hearing Initiative (AFHI)')
    target_audience = st.session_state.app_target_audience
    if target_audience == 'UX':
      release_name = 'UX release'
    elif target_audience == 'NAL':
      release_name = 'NAL release'
    else: # Default case, including 'ALL'.
      release_name = 'Main release'
    st.write(f'{common.DEMO_UPDATED} - {release_name}')
    st.write('For use with the Chrome browser on a '
             'desktop/laptop with your regular headphones.')
    st.write('**IMPORTANT:** Calibrate by setting your system volume to 50%. ')
    st.write('(For advanced calibration, see the Tone Generator & '
             'Calibration demo.)')

    # Add radio buttons to select the demo and toggle the current demo in
    # the session state based on the user's choice.
    demo_index = DEMOS.index(st.session_state.current_demo)
    demo_choice = st.radio(
      '**Select a demo:**',
      DEMOS,
      index=demo_index)
    # Switch demo if the user clicked a different one.
    if demo_choice != st.session_state.current_demo:
      reset_session_state()
      st.session_state.current_demo = demo_choice
      st.rerun()
    # Add some space before the reset button.
    st.markdown('')
    if st.button('Reset selected demo', type='primary'):
      reset_session_state()
      st.rerun()

    st.markdown('---')
    st.caption(
        '**Disclaimer:** The hearing tests in this application are '
        'being evaluated for equivalence to clinical audiometry. '
        'However, results depend on your audio hardware and acoustic '
        'environment, and should not be used for medical diagnosis.'
    )

def set_initial_session_state():
  """Sets the initial session state variables for the web app."""
  st.session_state.current_demo = 'Pure-Tone Audiometry'  # Default demo.
  st.session_state.initial_state_set = True
  st.session_state.app_target_audience = common.get_target_audience()
  # Determine if running locally based on environment variable.
  # Default to 'cloud', so nothing needs to be set in Streamlit Cloud.
  app_env = os.environ.get('APP_ENVIRONMENT', 'cloud').upper()
  st.session_state.is_running_locally = app_env == 'LOCAL'
  print(f'App environment: {app_env}, Running locally: '
        f'{st.session_state.is_running_locally}')

def main():
  """Runs the main web app containing the AFHI demos."""
  if 'initial_state_set' not in st.session_state:
    st.set_page_config(page_title='AFHI Web Demos',
                       page_icon=':ear:',
                       menu_items={})
    set_initial_session_state()
  load_css(CSS_PATH)
  create_sidebar()
  st.session_state.audio_container = st.container(height=50, border=False)
  with st.session_state.audio_container:
    st.empty()  # Create an empty element to occupy the initial space.
  # Run the selected demo.
  if st.session_state.current_demo in DEMO_FUNCTIONS:
    DEMO_FUNCTIONS[st.session_state.current_demo]()

if __name__ == '__main__':
  main()  # pragma: no cover
