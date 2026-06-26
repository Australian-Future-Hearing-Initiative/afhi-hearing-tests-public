"""Functionality for the tone generator demo part of the Streamlit app."""

import numpy as np
import streamlit as st
import time
import os
import tempfile
from scipy.io import wavfile
import zipfile
import io

import calibration
import demo_pta
import common

STEP_SIZE_DBHL = 5
EXTRA_PAUSE_S = 2.0  # Extra wait time after the tone train, in s.
DEFAULT_DITHER_SPL = 15  # Default dither noise level in dB SPL.
VCV_NOISE_FILE_PATH = os.path.join(
    common.PREFERRED_STIMULI_DIR, 'calibration', 'vcvs_calstim.wav')


def create_intro_text():
  """Creates the introductory text for the VCV synthesis demo."""
  st.title('Tone Generator')
  st.write('This allows you to generate all the tones in the pure-tone '
           'audiometry test. This is useful for debugging and calibration.')

def set_initial_demo_state():
  st.session_state.data_type = demo_pta.DEFAULT_DATATYPE
  st.session_state.tone_generator_intial_state_set = True
  st.session_state.tone_gen_vcv_noise_playing = False

def _play_short_silence():
  """Generates and plays a very short silent audio clip.
  
  This is used to stop the VCV noise when the button is pressed.
  """
  duration_s = 0.01
  num_samples = int(demo_pta.FS_HZ * duration_s)
  silence = np.zeros(num_samples, dtype=np.int16)
  stereo_silence = np.repeat(silence[:, np.newaxis], 2, axis=1)
  try:
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpf:
      wavfile.write(tmpf.name, demo_pta.FS_HZ, stereo_silence)
      common.autoplay_audio(tmpf.name)
  except Exception as e: # pylint: disable=broad-exception-caught
    st.error(f'Error playing silence: {e}')
    print(f'ERROR: Could not play silence file: {e}')

def show_options():
  """Displays the options for the tone generator demo."""
  # Create some radio buttons for the user to select the datatype.
  st.subheader('Settings')
  st.write('The default settings are those used in the PTA test, i.e., do '
           'not change them if you are simulating PTA test performance.')
  possible_data_types = ['float32', 'int32', 'int16']
  data_type = st.radio(
      'Data Type:',
      options=possible_data_types,
      index=possible_data_types.index(st.session_state.data_type),
      help='The datatype use to represent the tones.'
  )
  st.session_state.data_type = data_type
  # Add a toggle to add white dither noise to the tones. Directly set the
  # default here, because dither noise is not yet supported in the PTA test.
  help_msg_dither = ('Add white noise to the tones. This may improve the '
                     'perceived quality at low volumes.')
  add_dither_noise = st.toggle(
      f'Add {DEFAULT_DITHER_SPL} dB SPL dither noise',
      value=False,
      help=help_msg_dither,
  )
  st.session_state.tone_gen_add_dither_noise = add_dither_noise
  # Add a toggle to add an extra pause after each tone.
  help_msg_pause = ('Add an extra pause after the tone train. This may prevent '
                    'the tones from being truncated.')
  add_extra_pause = st.toggle(
      'Extend pause after tone',
      value=False,
      help=help_msg_pause,
  )
  st.session_state.tone_gen_add_extra_pause = add_extra_pause
  # Add a toggle to save the output locally.
  save_output_locally = st.toggle(
      'Save output locally',
      value=False,
      help='If checked, save the generated tone to a local file.',
  )
  st.session_state.tone_gen_save_output_locally = save_output_locally
  # Add a toggle for perceptual calibration.
  apply_cal = st.toggle(
      'Apply perceptual calibration',
      value=True,
      help='When true, map from dB HL to dB SPL using calibration data.'
  )
  st.session_state.tone_gen_apply_perceptual_calibration = apply_cal


def export_all_tones():
  """Generates all PTA tones with unit amplitude and zips them."""
  frequencies = sorted(set(common.STANDARD_FREQS_HZ))
  zip_buffer = io.BytesIO()

  with tempfile.TemporaryDirectory() as temp_dir:
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
      for freq in frequencies:
        # Define a unique path for each tone in the temporary directory.
        filename = f'tone_{freq}hz_unit_amp.wav'
        save_path = os.path.join(temp_dir, filename)

        # Generate the tone with unit amplitude and save it without playing.
        demo_pta.play_pulsed_tone(
            frequency_hz=freq,
            amplitude=1.0,
            save_path=save_path,
            play_audio=False,
            datatype='float32',  # Use float for max precision.
            dither_db_spl=-np.inf  # Ensure no noise is added.
        )
        # Add the generated file to the zip archive.
        zip_file.write(save_path, arcname=filename)
  zip_buffer.seek(0)
  return zip_buffer.getvalue()


def show_exports():
  """Displays the section for exporting tones."""
  st.subheader('Export Unit-Amplitude WAVs')
  st.write(
      'Use the button below to generate a zip file containing all the standard '
      'PTA test tones. Each tone is generated with unit amplitude (i.e., '
      'max value of 1.0) to be scaled by the receiving application.'
  )
  if st.button('Export All Tones',
               key='export_tones_button'):
    zip_bytes = export_all_tones()
    st.download_button(
        label='Download Tones.zip',
        data=zip_bytes,
        file_name='unit_amplitude_tones.zip',
        mime='application/zip'
    )


def show_calibration():
  """Displays the calibration section with the calibration tone button."""
  st.subheader('Calibration')
  st.write('Use the buttons below for calibration or level checking.')
  cal_spl = calibration.CALIBRATION_TONE_SPL

  col1, col2, _ = st.columns(3)

  with col1:
    if st.button(f'Play/stop {cal_spl} dB tone',
                 key='cal_tone_button', type='primary'):
      calibration.calibrate_audio()

  with col2:
    if st.button('Play/stop VCV noise', key='noise_button', type='primary'):
      if not st.session_state.tone_gen_vcv_noise_playing:
        if os.path.exists(VCV_NOISE_FILE_PATH):
          common.play_scaled_vcv_wav(
              VCV_NOISE_FILE_PATH,
              target_db_spl=calibration.DEFAULT_VCV_DB_SPL,
              ref_db_spl=calibration.REFERENCE_VCV_DB_SPL
          )
          st.session_state.tone_gen_vcv_noise_playing = True
        else:
          st.error(f'Noise file not found: {VCV_NOISE_FILE_PATH}')
          st.session_state.tone_gen_vcv_noise_playing = False
      else:
        _play_short_silence() # Call helper to play silence.
        st.session_state.tone_gen_vcv_noise_playing = False

def generate_button_grid():
  """Generates an array of buttons matching the tones needed in the PTA test."""
  st.subheader('Play tones')
  st.write('Click a button below to play the corresponding tone.')
  # Get the frequencies of the tones, making sure they are unique and sorted.
  frequencies = sorted(set(common.STANDARD_FREQS_HZ))
  # Get the amplitudes of the tones in dB HL.
  amplitudes_dbhl = np.arange(common.PTA_MIN_LEVEL_DB_HL,
                              common.PTA_MAX_LEVEL_DB_HL + 1, STEP_SIZE_DBHL)
  # Create a grid of buttons with frequencies across the columns and decreasing
  # amplitudes down the rows.
  n_cols = len(frequencies)
  n_rows = len(amplitudes_dbhl)
  # Create header (first row) with frequency labels.
  header_cols = st.columns(n_cols)
  for col_ind, col in enumerate(header_cols):
    frequency = frequencies[col_ind]
    # Add frequency label to header.
    col.markdown(f'**{frequency} Hz**')
  button_css = """
  <style>
  """
  row_offset = 19  # Accounts for other Streamlit elements above the buttons.
  for row_ind in range(n_rows):
    # Calculate a slight color variation for each row.
    hue_shift = row_ind * 16  # Adjust to change color variation speed.
    # Hue from green/blue (quieter) to red (louder)
    background_hue = 220 - hue_shift  # Blue-ish (hue 240) to red (hue 0).
    if background_hue < 0:
      background_hue = 0
    background_color = f'hsl({background_hue}, 60%, 80%)'
    button_css += f"""
    div.stHorizontalBlock:nth-child({row_ind + row_offset})
    div.stButton > button[kind="secondary"] {{
        width: 70px;
        height: 50px;
        background-color: {background_color};
        color: black; /* Or white, depending on the background color */
    }}
    """
  button_css += """
  </style>
  """
  st.markdown(button_css, unsafe_allow_html=True)

  for row_ind in range(n_rows):
    cols = st.columns(n_cols)
    for col_ind, col in enumerate(cols):
      frequency = frequencies[col_ind]
      amplitude_dbhl = amplitudes_dbhl[row_ind]
      # Create a button with the amplitude label.
      if col.button(f'{amplitude_dbhl}\ndBHL',
                    key=f'{frequency}_{amplitude_dbhl}'):
        if st.session_state.tone_gen_apply_perceptual_calibration:
          dbspl = calibration.dbhl_to_dbspl(
              amplitude_dbhl, frequency, 'PixelBuds_HughsonWestlake')
        else:
          dbspl = amplitude_dbhl
        tone_amp = calibration.dbspl_to_amp(dbspl)
        if st.session_state.tone_gen_add_dither_noise:
          dither_db_spl = DEFAULT_DITHER_SPL
        else:
          dither_db_spl = None
        if st.session_state.tone_gen_save_output_locally:
          output_dir = 'tone_generator_output'
          if not os.path.exists(output_dir):
            os.makedirs(output_dir)
          save_path = os.path.join(
              output_dir, f'tone_{frequency}Hz_{amplitude_dbhl}dBHL.wav')
        else:
          save_path = None
        demo_pta.play_pulsed_tone(
          frequency,
          tone_amp,
          datatype=st.session_state.data_type,
          dither_db_spl=dither_db_spl,
          save_path=save_path)
        # The default pause is the same as used in the PTA test.
        time.sleep(demo_pta.RESPONSE_WINDOW_S)
        # Add the optional extra pause.
        if st.session_state.tone_gen_add_extra_pause:
          time.sleep(EXTRA_PAUSE_S)
        st.rerun()

def create_main_demo():
  """Creates the main tone generator demo."""
  if 'tone_generator_intial_state_set' not in st.session_state:
    set_initial_demo_state()
  create_intro_text()
  show_calibration()
  st.session_state.calibration_playing = False
  show_options()
  generate_button_grid()
  show_exports()
