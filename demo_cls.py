"""Functionality for the categorical loudness scaling demo."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
import streamlit as st
import tempfile
import time
from datetime import datetime

import audio_tools
import calibration
import cls_results
import common
import run_qcls
import hearing_models

# Set up mapping of buttons, button labels, and button numbers.
BUTTON_MAPPING = [('extremely_loud', 'Extremely Loud', 10),
                  ('very_loud', 'Very Loud', 9),
                  ('loud_to_very_loud', '', 8),
                  ('loud', 'Loud', 7),
                  ('medium_to_loud', '', 6),
                  ('medium', 'Medium', 5),
                  ('soft_to_medium', '', 4),
                  ('soft', 'Soft', 3),
                  ('very_soft_to_soft', '', 2),
                  ('very_soft', 'Very Soft', 1),
                  ('not_heard', 'Not Heard', 0)
                  ]
FS_HZ = 44100  # Sampling frequency in Hz used to synthesize the tones.
TONE_DURATION_S = 0.7
# Silence prepended to each tone so the spinner renders
# in the browser before the audible portion begins.
TONE_LEAD_SILENCE_S = 0.4
TEST_FREQS_KHZ = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8]
DEFAULT_N_TESTS = 80  # Default number of samples per ear.
MAX_N_TESTS = 160  # Maximum number of samples per ear.
INITIAL_MAX_VOLUME_DB_SPL = 80  # Initial max volume in dB SPL.
NOISE_REDUCTION_IF_EXTREMELY_LOUD = 10  # dB reduction for 'Extremely Loud'.
RESPONSE_PROMPT = 'How loud was the sound you just heard?'
# Practice tones played before the real test begins.
# Binaural, 1 kHz, descending levels (loud to soft).
PRACTICE_FREQ_HZ = 1000
PRACTICE_LEVELS_DB_SPL = [65, 50, 35]
PRACTICE_STIMULI = [
    ('both', PRACTICE_FREQ_HZ,
     calibration.dbspl_to_amp(lvl))
    for lvl in PRACTICE_LEVELS_DB_SPL
]
# Initial model parameters for each ear.
INITIAL_MODEL = {
  'component_coeffs': np.asarray([0.01, 0]),
  'sone_intersection': 24
}

def set_initial_demo_state():
  """Sets initial session state variables specific to the CLS test."""
  st.session_state.cls_responses = []
  st.session_state.cls_stimuli = []
  st.session_state.cls_n_tests = DEFAULT_N_TESTS  # Number of samples per ear.
  st.session_state.cls_state = 'Initial'  # 'Initial', 'Running', 'Completed'.
  st.session_state.cls_stimulus_state = None  # None, 'Playing', 'Waiting'.
  st.session_state.cls_initial_state_set = True
  st.session_state.cls_max_volume = INITIAL_MAX_VOLUME_DB_SPL
  st.session_state.cls_tone_start_time = None  # Start time of the current tone.
  st.session_state.cls_start_time = None  # Start time of the test.
  st.session_state.cls_duration_s = None  # Duration of the entire test.
  st.session_state.cls_merge_lr = False  # Default value for binaural settings.
  st.session_state.cls_volume_confirmed = False  # Track volume confirmation.
  # Initialize separate models for each ear.
  st.session_state.cls_loudness_model_left = INITIAL_MODEL.copy()
  st.session_state.cls_loudness_model_right = INITIAL_MODEL.copy()
  st.session_state.cls_batch_number = 0  # For learning rate calculation.
  # Initialize ear sequence tracking.
  st.session_state.cls_ear_sequence = []  # Will be populated when test starts.
  st.session_state.cls_current_sequence_index = 0  # Track position in sequence.
  st.session_state.cls_response_pending = False  # Double-click guard.
  st.session_state.cls_in_practice = False  # Practice phase flag.
  st.session_state.cls_practice_index = 0  # Current practice tone.
  st.session_state.cls_backup_saved = False # Flag for local backup status.
  # Sequential ear testing state.
  st.session_state.cls_current_ear_phase = None  # Current ear being tested.
  st.session_state.cls_first_ear = None  # Starting ear (always left).
  st.session_state.cls_archive_stimuli = []  # First ear's stimuli.
  st.session_state.cls_archive_responses = []  # First ear's responses.

def format_model_params(model: dict, ear: str) -> str:
  """Formats the loudness model parameters into a readable string."""
  coeffs = model['component_coeffs']
  sone = model['sone_intersection']
  # Use np.array2string for consistent formatting of the numpy array.
  coeffs_str = np.array2string(coeffs, precision=4, separator=', ')
  return (f'Loudness Model Parameters ({ear} ear):\n'
          f'--------------------------------------\n'
          f'Component Coefficients: {coeffs_str}\n'
          f'Sone Intersection: {sone:.4f}\n')

def create_intro_text():
  """Creates the introductory text for the CLS demo."""
  st.title('Categorical Loudness Scaling')
  st.write('Categorical loudness scaling (CLS) is a method for assessing an '
           "individual's perception of loudness across a range of frequencies "
           'and sound levels. This test plays a series of tones at various '
           'volumes and asks you to categorise how loud each tone sounds to '
           'you. The results are used to create a personalised profile of your '
           'loudness perception, informing the fitting of hearing aids.')

def play_tone(
    ear: str, frequency_hz: float, volume: float, duration_s: float):
  """Generates and plays a tone to the user.

  Args:
    ear: The ear to play the tone in ('left', 'right', or 'both').
    frequency_hz: The frequency of the tone in Hz.
    volume: The volume of the tone, between 0 and 1.
    duration_s: The duration of the tone in seconds.
  """
  # Ensure the volume is between 0 and 1.
  if volume < 0 or volume > 1:
    raise ValueError('Volume must be between 0 and 1.')
  # Generate the tone.
  t = np.linspace(0, duration_s, int(FS_HZ * duration_s))
  tone = np.sin(2 * np.pi * frequency_hz * t)
  # Apply a Tukey window to the start/end of the pulse to avoid start click.
  tone = audio_tools.tukey_window(tone, 0.1)
  tone = tone * volume  # Adjust the volume.
  # Convert to 32-bit int format.
  tone = (tone * common.MAX_32_BIT_INT).astype(np.int32)
  # Create a stereo tone and mute the appropriate channel, if needed.
  tone = np.repeat(tone[:, np.newaxis], 2, axis=1)
  if ear == 'left':
    tone[:, 1] = 0  # Mute right channel.
  elif ear == 'right':
    tone[:, 0] = 0  # Mute left channel.
  # Prepend silence so the spinner has time to render
  # in the browser before the audible tone begins.
  n_silent = int(FS_HZ * TONE_LEAD_SILENCE_S)
  silence = np.zeros((n_silent, 2), dtype=tone.dtype)
  tone = np.concatenate([silence, tone], axis=0)
  # Play the tone and show a spinner to indicate playback.
  extra_delay_s = 0.8
  with st.session_state.spinner_placeholder, \
       st.spinner('Playing ...'):
    # Save the tone to a temporary WAV file.
    with tempfile.NamedTemporaryFile(
        suffix='.wav', delete=False
    ) as temp_file:
      wavfile.write(temp_file.name, FS_HZ, tone)
      # Play the temporary file.
      common.autoplay_audio(temp_file.name)
    # Pause for audio to play. This is necessary to
    # avoid audio glitches. The extra delay beyond the
    # tone duration seems especially critical for
    # bluetooth headphones (with higher latency).
    time.sleep(duration_s + extra_delay_s)

def generate_next_stimulus(
    ear: str) -> tuple[str, float, float]:
  """Generates parameters for the next tone using the adaptive model.

  Uses the loudness model (fitted to previous responses) to
  select the next stimulus frequency and amplitude.

  Args:
    ear: The ear to generate the stimulus for
      ('left', 'right', or 'both').

  Returns:
    A tuple containing the ear ('left', 'right', or
    'both'), frequency (in Hz), and volume (linear
    amplitude) in the range [0, 1].
  """
  # Follow the approach from run_qcls.py for frequency range.
  response_count = len(st.session_state.cls_responses)
  if response_count < 10:  # First batch focuses on important range.
    min_freq = 1000
    max_freq = 4000
  else:  # Subsequent batches expand to full range.
    min_freq = min(TEST_FREQS_KHZ) * 1000
    max_freq = max(TEST_FREQS_KHZ) * 1000
  # Use the appropriate ear-specific model.
  if ear == 'left' or ear == 'both':
    model = st.session_state.cls_loudness_model_left
  else:  # Right ear.
    model = st.session_state.cls_loudness_model_right
  # Use the hearing model to generate test frequencies
  # and amplitudes.
  freq_hz, amplitude = (
      hearing_models.random_test_frequency_and_amplitude(
          min_freq=min_freq,
          max_freq=max_freq,
          loudness_model=model
      )
  )
  # Make sure amplitude does not exceed the maximum allowed.
  max_amp = calibration.dbspl_to_amp(st.session_state.cls_max_volume)
  if amplitude > max_amp:
    amplitude = max_amp
    print('Reduced amplitude to:', amplitude)
  # Clip the amplitude to the [0, 1] range.
  amplitude = np.clip(amplitude, 0, 1)
  print(f'Ear: {ear}, Amplitude: {amplitude}, Frequency: {freq_hz}')
  return ear, freq_hz, amplitude

def start_button_click():
  """Handles the start/replay button click."""
  if st.session_state.cls_state == 'Initial':
    # Hadn't started the test before, so start now.
    st.session_state.cls_start_time = time.time()
    st.session_state.cls_state = 'Running'
    # Create the ear sequence based on slider value and merge setting.
    samples_per_ear = st.session_state.cls_n_tests
    if st.session_state.cls_merge_lr:
      st.session_state.cls_ear_sequence = (
          ['both'] * samples_per_ear
      )
    else:
      # Sequential mode: complete one ear fully, then
      # the other. Always start with the left ear.
      first_ear = 'left'
      st.session_state.cls_first_ear = first_ear
      st.session_state.cls_current_ear_phase = (
          first_ear
      )
      st.session_state.cls_ear_sequence = (
          [first_ear] * samples_per_ear
      )
    st.session_state.cls_current_sequence_index = 0
    # Start the practice phase before the real test.
    st.session_state.cls_in_practice = True
    st.session_state.cls_practice_index = 0
    st.session_state.cls_stimuli.append(
        PRACTICE_STIMULI[0])
  st.session_state.cls_stimulus_state = 'Playing'
  st.rerun()

def response_button_click(button_number):
  """Handles the response button click."""
  # Guard against double-click processing.
  if st.session_state.get('cls_response_pending', False):
    return
  st.session_state.cls_response_pending = True
  # Clear the response prompt.
  st.session_state.response_prompt.empty()
  # Handle practice phase responses (discard, no model
  # update).
  if st.session_state.cls_in_practice:
    idx = st.session_state.cls_practice_index + 1
    if idx < len(PRACTICE_STIMULI):
      # Queue the next practice tone.
      st.session_state.cls_practice_index = idx
      st.session_state.cls_stimuli.append(
          PRACTICE_STIMULI[idx])
      st.session_state.cls_stimulus_state = 'Playing'
    else:
      # Practice complete; start the real test.
      st.session_state.cls_in_practice = False
      st.session_state.cls_stimuli.clear()
      st.session_state.cls_start_time = time.time()
      ear = st.session_state.cls_ear_sequence[0]
      st.session_state.cls_stimuli.append(
          generate_next_stimulus(ear=ear))
      st.session_state.cls_stimulus_state = 'Playing'
    st.rerun()
  # Calculate the response time.
  if st.session_state.cls_tone_start_time is not None:
    response_time_s = time.time() - st.session_state.cls_tone_start_time
    st.session_state.cls_tone_start_time = None
  else:
    response_time_s = 0
  # Check if the button was the 'Extremely Loud' button and, if so, reduce
  # the max volume by a fixed amount. (This is not related to the previous
  # tone volume, due to the risk that the button was pressed by accident.)
  if button_number == 10:
    st.session_state.cls_max_volume -= NOISE_REDUCTION_IF_EXTREMELY_LOUD
    print('Reduced max volume to:', st.session_state.cls_max_volume)
  # Record the response with response time.
  st.session_state.cls_responses.append((button_number, response_time_s))

  # Update the loudness model based on the latest response.
  ear, frequency_hz, amplitude = st.session_state.cls_stimuli[-1]
  # Convert button number to categorical units.
  cu = hearing_models.BUTTONS_TO_CUS[button_number]
  # Calculate learning rate that decreases with more responses.
  # Using the same formula as in run_qcls.py.
  st.session_state.cls_batch_number += 1
  learning_rate = 1.25 / (st.session_state.cls_batch_number + 1)
  # Update the appropriate model(s) using the latest response.
  print('\nUpdating model with the following: ')
  print(f'Ear: {ear}')
  print(f'Frequency: {frequency_hz}, \nAmplitude: {amplitude}, \nCU: {cu}')

  # Select which model to update.
  old_model = (st.session_state.cls_loudness_model_left
              if ear in ['left', 'both']
              else st.session_state.cls_loudness_model_right)
  # Get an updated version of the model.
  updated_model, _ = hearing_models.update_loudness_model(
      frequencies=np.array([frequency_hz]),
      amplitudes=np.array([amplitude]),
      cus=np.array([cu]),
      loudness_model=old_model,
      rate=learning_rate
  )
  # Copy the updated model back to the appropriate model(s).
  if ear in ['left', 'both']:
    st.session_state.cls_loudness_model_left = updated_model
    print('Updated left ear model:', updated_model)
  if ear in ['right', 'both']:
    st.session_state.cls_loudness_model_right = updated_model
    print('Updated right ear model:', updated_model)

  # Check if the current ear phase is complete.
  st.session_state.cls_current_sequence_index += 1
  if st.session_state.cls_current_sequence_index >= len(
      st.session_state.cls_ear_sequence):
    # Determine whether to transition to the second
    # ear or mark the test as completed.
    first_ear = st.session_state.cls_first_ear
    current_phase = (
        st.session_state.cls_current_ear_phase
    )
    need_second_ear = (
        not st.session_state.cls_merge_lr
        and current_phase == first_ear
    )
    if need_second_ear:
      # Archive the first ear's data.
      st.session_state.cls_archive_stimuli = (
          list(st.session_state.cls_stimuli)
      )
      st.session_state.cls_archive_responses = (
          list(st.session_state.cls_responses)
      )
      # Reset adaptive state for the second ear.
      second_ear = (
          'right' if first_ear == 'left'
          else 'left'
      )
      n = st.session_state.cls_n_tests
      st.session_state.cls_current_ear_phase = (
          second_ear
      )
      st.session_state.cls_ear_sequence = (
          [second_ear] * n
      )
      st.session_state.cls_current_sequence_index = 0
      st.session_state.cls_batch_number = 0
      st.session_state.cls_max_volume = (
          INITIAL_MAX_VOLUME_DB_SPL
      )
      st.session_state.cls_stimuli = []
      st.session_state.cls_responses = []
      # Generate the first stimulus for the new ear.
      st.session_state.cls_stimuli.append(
          generate_next_stimulus(ear=second_ear)
      )
      st.session_state.cls_stimulus_state = 'Playing'
    else:
      # Test fully completed (merge_lr or second ear).
      st.session_state.cls_state = 'Completed'
      st.session_state.cls_duration_s = (
          time.time()
          - st.session_state.cls_start_time
      )
      st.session_state.cls_stimulus_state = None
      st.session_state.cls_response_pending = False
  else:
    # Continue within the current ear phase.
    seq_idx = (
        st.session_state.cls_current_sequence_index
    )
    ear = (
        st.session_state.cls_ear_sequence[seq_idx]
    )
    st.session_state.cls_stimuli.append(
        generate_next_stimulus(ear=ear)
    )
    st.session_state.cls_stimulus_state = 'Playing'
  st.rerun()

def create_progress_bar():
  """Creates a progress bar for the CLS test."""
  if (st.session_state.cls_state != 'Initial'
      and not st.session_state.get(
          'cls_in_practice', False)):
    # Calculate total progress across both ear phases.
    if st.session_state.cls_merge_lr:
      total = len(
          st.session_state.cls_ear_sequence
      )
      done = (
          st.session_state
          .cls_current_sequence_index
      )
    else:
      total = st.session_state.cls_n_tests * 2
      done = (
          len(
              st.session_state
              .cls_archive_stimuli
          )
          + st.session_state
          .cls_current_sequence_index
      )
    progress = min(1.0, done / total)

    st.progress(progress)
    st.write(
        f'{int(progress * 100)}% Complete'
    )

def show_cls_test_buttons():
  """Displays the response buttons for the CLS test."""
  st.write('\n')
  st.subheader('Take the test')
  # Volume confirmation checkbox.
  help_str = 'You must confirm your volume setting before starting the test.'
  st.checkbox('I confirm my volume is set to 50%',
             key='cls_volume_confirm',
             value=st.session_state.cls_volume_confirmed,
             disabled=st.session_state.cls_state != 'Initial',
             help=help_str,
             on_change=lambda: st.session_state.update(
                 {'cls_volume_confirmed': st.session_state.cls_volume_confirm}))

  # Start/Replay button.
  if st.button(
      'Start/Replay', key='cls_start_replay',
      disabled=(
        not st.session_state.cls_volume_confirmed
        or st.session_state.cls_stimulus_state
        == 'Playing'
        or st.session_state.cls_state
        == 'Completed'),
      icon=':material/play_arrow:'):
    start_button_click()
  # Phone area: a positioning wrapper that holds both
  # the spinner overlay and the button phone layout.
  with st.container(key='cls_phone_area'):
    # Spinner overlaid on the phone centre via CSS.
    with st.container(key='cls_spinner'):
      st.session_state.spinner_placeholder = (
        st.empty()
      )
    # Buttons in a keyed container so styles.css can
    # scope gap-collapse and bezel styling.
    with st.container(key='cls_buttons'):
      for key, label, number in BUTTON_MAPPING:
        if st.button(
            label=label, key=key,
            disabled=(
              st.session_state.cls_stimulus_state
              != 'Waiting'
            )):
          response_button_click(number)
  # Response prompt below the phone. Wrapped in a
  # fixed-height container so the progress bar does
  # not jump when the text appears/disappears.
  with st.container(height=45, border=False):
    st.session_state.response_prompt = st.empty()
    if st.session_state.cls_stimulus_state == 'Waiting':
      if st.session_state.cls_in_practice:
        n = st.session_state.cls_practice_index + 1
        total = len(PRACTICE_STIMULI)
        prompt = (
            f'Practice ({n} of {total})'
            f' — {RESPONSE_PROMPT}'
        )
      else:
        prompt = RESPONSE_PROMPT
      st.session_state.response_prompt.markdown(
        f'<p style="text-align: center;">'
        f'{prompt}</p>',
        unsafe_allow_html=True
      )
  create_progress_bar()

def display_settings():
  """Displays the settings for the CLS test."""
  st.subheader(common.SETTINGS_TITLE)
  st.write(common.SETTINGS_STRING)
  # Disable all settings once test has started.
  is_nal = st.session_state.get('app_target_audience') == 'NAL'
  settings_disabled = (st.session_state.cls_state == 'Running' or
                      st.session_state.cls_state == 'Completed' or
                      is_nal)
  st.toggle('Merge L/R', key='toggle_cls_merge_lr',
            help=common.MERGE_LR_HELP,
            disabled=settings_disabled)
  if st.session_state.toggle_cls_merge_lr:
    st.session_state.cls_merge_lr = True
  else:
    st.session_state.cls_merge_lr = False
  st.session_state.cls_n_tests = st.slider(
    'Set the number of samples per ear:',
    1, MAX_N_TESTS, st.session_state.cls_n_tests,
    disabled=settings_disabled)

def create_main_demo():
  """Controls the states and state transitions of the main demo."""
  if 'cls_initial_state_set' not in st.session_state:
    set_initial_demo_state()
  # Display material that is present for all states.
  create_intro_text()
  display_settings()
  common.display_preparation()
  show_cls_test_buttons()
  # Take actions specific to the current state.
  if st.session_state.cls_state == 'Running':
    if st.session_state.cls_stimulus_state == 'Playing':
      ear, frequency_hz, volume = (
          st.session_state.cls_stimuli[-1])
      st.session_state.cls_tone_start_time = time.time()
      play_tone(ear, frequency_hz, volume, TONE_DURATION_S)
      st.session_state.cls_stimulus_state = 'Waiting'
      st.session_state.cls_response_pending = False
      st.rerun()
  if st.session_state.cls_state == 'Completed':
    st.subheader('Results')
    st.write(
        'The inferred thresholds for each ear '
        'are shown below:'
    )
    # Combine archived (first ear) and current
    # (second ear) data. For merge_lr the archive
    # lists are empty, so this is a no-op concat.
    all_stimuli = (
        st.session_state.cls_archive_stimuli
        + st.session_state.cls_stimuli
    )
    all_responses = (
        st.session_state.cls_archive_responses
        + st.session_state.cls_responses
    )
    ear = [e for e, f, v in all_stimuli]
    freq = [f for e, f, v in all_stimuli]
    vol = [v for e, f, v in all_stimuli]
    responses = all_responses
    # Guard against state corruption (e.g. concurrent
    # reruns on Streamlit Cloud).
    if len(ear) != len(responses):
      st.error(
          f'Data integrity error: {len(ear)} stimuli '
          f'vs {len(responses)} responses. '
          'Please restart the test.'
      )
      return
    # Make a dataframe with header freq, vol, response.
    df = pd.DataFrame({'Ear': ear,
                      'Frequency (Hz)': freq,
                      'Amplitude': vol,
                      'Button': [r[0] for r in responses],
                      'Response Time (s)': [r[1] for r in responses]})
    df.index.name = 'Stimulus'
    df.index += 1  # Make the index start at 1.
    # Plot results in stereo (will be identical if 'Merge L/R' is on).
    res_fig = run_qcls.plot_qcls_results_stereo(
        df[df['Ear'].isin(['left', 'both'])],
        st.session_state.cls_loudness_model_left,
        df[df['Ear'].isin(['right', 'both'])],
        st.session_state.cls_loudness_model_right)
    if res_fig is not None:
      st.pyplot(res_fig)
      # Prepare data for download/email.
      test_name = 'Categorical Loudness Scaling'
      zip_prefix = 'cls_results'
      csv_content = cls_results.generate_cls_results_csv(
          df, st.session_state.cls_duration_s
      )
      # Format model parameters.
      left_model_str = format_model_params(
          st.session_state.cls_loudness_model_left, 'Left')
      right_model_str = format_model_params(
          st.session_state.cls_loudness_model_right, 'Right')
      # Define files for zip.
      files_for_zip = [
          ('cls_full_results.csv', csv_content),
          ('cls_threshold.png', res_fig),
          ('model_params_left.txt', left_model_str),
          ('model_params_right.txt', right_model_str)
      ]

      # Generate zip data and filename.
      zip_data = common.generate_zip_bytes(files_for_zip)
      timestamp = datetime.now().strftime('%Y%m%d_%H%M')
      zip_filename = f'UTC{timestamp}_{zip_prefix}.zip'

      # Save local backup if applicable; set flag to prevent multiple backups.
      if (st.session_state.is_running_locally and
          st.session_state.app_target_audience == 'NAL' and
          not st.session_state.get('cls_backup_saved', False)):
        common.save_local_backup(zip_data, zip_filename)
        st.session_state.cls_backup_saved = True
        print('CLS local backup saved.')

      st.write('\n\n')  # Add some space before the download button.
      st.download_button(
          label='Download results',
          data=zip_data,
          file_name=zip_filename,
          mime='application/zip',
          key='cls_manual_download'
      )
      common.display_email_results_form(test_name, files_for_zip, zip_prefix)

      # Close the figure after potential use in zip/email.
      plt.close(res_fig)
      # Conditionally display interpretation.
      if st.session_state.app_target_audience != 'NAL':
        cls_results.display_interpretation()
