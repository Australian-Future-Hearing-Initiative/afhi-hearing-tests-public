"""Functionality for the pure-tone audiometry (PTA) demo."""

import time
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

import audio_tools
import audiogram
import calibration
import common
import pta_results
import pta_algorithms


FS_HZ = 44100  # Sampling frequency in Hz used to synthesize the tones.
N_PULSES = 3  # Number of pulses in the tone train.
PULSE_DURATION_S = 0.2  # Duration of each pulse in seconds.
GAP_DURATION_S = 0.25  # Duration of the gap between pulses in seconds.
DEMO_TONE_FREQ_HZ = 500  # Frequency of the demo tone in Hz.
DEMO_TONE_AMP_DBHL = 65  # Amplitude of the demo tone in dB HL.
FREQS_HZ = common.STANDARD_FREQS_HZ  # Test frequencies.
DEFAULT_STARTING_OFFSET_DBHL = 10  # Default offset for starting level.
CACHE_BUSTING_NOISE_DB_SPL = 0.1  # Level of noise to add to prevent caching.
# Set the min and max levels for the test in dBHL. These are close to the
# conventional levels, but are slightly adjusted to avoid the extremes and limit
# the dynamic range needed and test duration.
START_LEVEL_DB_HL = common.PTA_START_LEVEL_DB_HL
MIN_LEVEL_DB_HL = common.PTA_MIN_LEVEL_DB_HL
MAX_LEVEL_DB_HL = common.PTA_MAX_LEVEL_DB_HL
MAX_TONES_ADAPTIVE = common.PTA_MAX_TRIALS_PER_EAR
MIN_ONSET_TIME_S = 0.4  # Minimum value of random onset time in seconds.
MAX_ONSET_TIME_S = 1.0  # Maximum value of random onset time in seconds.
RESPONSE_WINDOW_S = 4.0  # Response window after the start of tone in seconds.
DEFAULT_DATATYPE = 'float32'
# Define the names of the two approaches.
BASIC_METHOD_NAME = 'Hughson-Westlake (older, slower)'
ADVANCED_METHOD_NAME = 'Adaptive PTA (newer, faster)'


def set_initial_demo_state():
  """Sets initial session state variables specific to the PTA test."""
  st.session_state.pta_method = ADVANCED_METHOD_NAME   # Default method.
  st.session_state.pta_state = 'Initial'  # 'Initial', 'Running', 'Completed'.
  st.session_state.pta_merge_lr = False  # Default value for binaural settings.
  if st.session_state.pta_merge_lr:
    st.session_state.pta_current_ear = 'both'
  else:
    st.session_state.pta_current_ear = common.DEFAULT_INITIAL_EAR
  st.session_state.pta_current_freq_ind = 0  # Start with the first frequency.
  st.session_state.pta_current_db_hl = START_LEVEL_DB_HL
  st.session_state.pta_results = []  # List for storing results.
  st.session_state.pta_thresholds_left = {}  # Dict to store left thresholds.
  st.session_state.pta_thresholds_right = {}  # Dict to store right thresholds.
  st.session_state.pta_thresholds_both = {}  # Dict to store merged thresholds.
  st.session_state.pta_using_canned_data = False  # Indicate canned data loaded.
  st.session_state.pta_in_wait_period = False  # Indicate if in wait period.
  st.session_state.pta_start_time = None # Start time of the test.
  st.session_state.pta_duration_s = None  # Duration of the test in seconds.
  st.session_state.pta_tone_start_time = None  # Start time of the current tone.
  st.session_state.pta_initial_state_set = True
  st.session_state.pta_backup_saved = False # Flag for local backup status.

def create_intro_text():
  """Creates the introductory text for the pure-tone audiometry demo."""
  st.title('Pure-Tone Audiometry')
  st.write('Pure-tone audiometry is a hearing test that measures your '
           'threshold of hearing at various frequencies. This test plays a '
           'series of pulsed tones and asks you to indicate when you can hear '
           'the tone. The results are used to create an audiogram.')

def play_pulsed_tone(frequency_hz: float,
                     amplitude: float,
                     ear: str = 'both',
                     n_pulses: int = N_PULSES,
                     pulse_duration_s: float = PULSE_DURATION_S,
                     gap_duration_s: float = GAP_DURATION_S,
                     datatype: str = DEFAULT_DATATYPE,
                     dither_db_spl: float = None,
                     save_path: str = None,
                     play_audio: bool = True):
  """Generate and plays a pulsed tone to the user.

  Args:
    frequency_hz: The frequency of the tone in Hz.
    amplitude: The amplitude of the tone in range [0, 1].
    ear: The ear to play the tone in ('left', 'right', or 'both').
    n_pulses: The number of pulses in the tone train.
    pulse_duration_s: The duration of each pulse in seconds.
    gap_duration_s: The duration of the gap between pulses in seconds.
    datatype: The datatype of the audio samples ('float32', 'int32', 'int16').
    add_dither: Whether to add white dither noise to the tones.
    save_path: The path to save the generated tone to. If None, a temporary
      file will be used.
    play_audio: If False, the audio will not be played.
  """
  if amplitude < 0 or amplitude > 1:  # Ensure the volume is between 0 and 1.
    # Clip the amplitude to the valid range, but issue warning.
    st.warning('Amplitude must be between 0 and 1. Clipping to valid range.')
    amplitude = np.clip(amplitude, 0, 1)
  # Generate an individual pulse.
  t = np.linspace(0, pulse_duration_s, int(FS_HZ * pulse_duration_s))
  pulse = np.sin(2 * np.pi * frequency_hz * t)
  pulse = pulse * amplitude  # Adjust the volume.
  # Apply a Tukey window to the start/end of the pulse to avoid clicking or
  # 'sticky' sounds at start and end of the tone.
  pulse = audio_tools.tukey_window(pulse, 0.1)
  # Generate the tone train with gaps between pulses. Starting with a gap seems
  # to avoid weird behavior at the beginning of the tone train when playing in
  # a single ear.
  gaps = np.zeros(int(FS_HZ * gap_duration_s))
  tone = np.concatenate([gaps, pulse] * n_pulses)
  # Add white dither noise if enabled or a tiny amount of noise if not.
  if dither_db_spl is None:
    # Add an imperceptible amount of noise anyway. This is to prevent the
    # web browser from attempting to replay the same tone and failing. The noise
    # will change the hash of the tone and prevent the browser from using the
    # cached version.
    noise_amplitude = calibration.dbspl_to_amp(CACHE_BUSTING_NOISE_DB_SPL)
  else:
    noise_amplitude = calibration.dbspl_to_amp(dither_db_spl)
  noise = np.random.randn(len(tone))
  tone = tone + noise_amplitude * noise
  if datatype == 'float32':
    tone = tone.astype(np.float32)
  elif datatype == 'int32':
    tone = (tone * common.MAX_32_BIT_INT).astype(np.int32)
  elif datatype == 'int16':
    tone = (tone * common.MAX_16_BIT_INT).astype(np.int16)
  else:
    raise ValueError('Invalid datatype. Must be float32, int32, or int16.')
  # Create a 2-channel stereo array.
  tone = np.repeat(tone[:, np.newaxis], 2, axis=1)
  # Zero out one channel if only one ear is selected.
  if ear == 'left':
    tone[:, 1] = 0  # Mute right ear.
  elif ear == 'right':
    tone[:, 0] = 0  # Mute left ear.
  # Save the tone to a temporary WAV file.
  if save_path:
    wavfile.write(save_path, FS_HZ, tone)
    if play_audio:
      common.autoplay_audio(save_path)
  else:
    with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
      wavfile.write(temp_file.name, FS_HZ, tone)
      if play_audio:
        common.autoplay_audio(temp_file.name)  # Play the temporary file.

def demo_button():
  """Displays the demo button and supporting info."""
  st.write('Use the buttons below to hear an example of a pulsed test tone '
           'in each ear.')
  col1, col2, _ = st.columns([1, 1, 1])
  dbspl = calibration.dbhl_to_dbspl(DEMO_TONE_AMP_DBHL,
                                    DEMO_TONE_FREQ_HZ,
                                    'PixelBuds_HughsonWestlake')
  dbspl += calibration.get_device_offset(
      DEMO_TONE_FREQ_HZ, st.session_state.pta_device)
  tone_amp = calibration.dbspl_to_amp(dbspl)
  with col1:
    if st.button('Play demo tone - L', key = 'pta_demo_tone_left',
                 icon=':material/play_arrow:'):
      play_pulsed_tone(DEMO_TONE_FREQ_HZ, tone_amp, ear='left')
      time.sleep(2)
      st.rerun()
  with col2:
    if st.button('Play demo tone - R', key = 'pta_demo_tone_right',
                 icon=':material/play_arrow:'):
      play_pulsed_tone(DEMO_TONE_FREQ_HZ, tone_amp, ear='right')
      time.sleep(2)
      st.rerun()

def start_button():
  """Displays the start button and handles its functionality."""
  if st.button('Start the test', key='pta_start_test',
               icon=':material/play_arrow:',
               disabled=st.session_state.pta_state == 'Running'):
    set_initial_demo_state()
    st.session_state.pta_state = 'Running'
    st.session_state.pta_start_time = time.time()
    st.rerun()

def cancel_button():
  """Displays the cancel button and handles its functionality."""
  if st.button('Cancel the test', key='pta_cancel_test',
               icon=':material/cancel:',
               disabled=st.session_state.pta_state != 'Running'):
    set_initial_demo_state()
    st.rerun()

def dbhl_to_amplitude(db_hl: float, max_db_hl=MAX_LEVEL_DB_HL) -> float:
  """Converts dB HL to a linear amplitude scale.

  Maps the dB HL values to a linear amplitude scale in the range [0, 1]. This is
  an arbitrary mapping (in the absence of any calibration) that is used to set
  the amplitude of the tones.

  Args:
    db_hl: The dB HL value to convert to amplitude.
    max_db_hl: The db HL value that should be mapped to an amplitude of 1.

  Returns:
    The amplitude value in the range [0, 1].

  Raises:
    ValueError: If the db HL value is greater than the max value.
  """
  if db_hl > max_db_hl:
    raise ValueError('db HL value must be less than or equal to the max value.')
  unscaled_amplitude = 10 ** (db_hl / 20)
  max_unscaled_amplitude = 10 ** (max_db_hl / 20)
  amplitude = unscaled_amplitude / max_unscaled_amplitude
  return amplitude

def hughson_westlake_next_step(frequency_hz, ear):
  """Runs a modified Hughson-Westlake procedure for a given frequency.

  The Hughson-Westlake procedure is an 'adaptive staircase' method that
  decreases  the intensity of the tone until the user can no longer hear it,
  and then increases it again until they can.

  Args:
    frequency_hz: The frequency of the tone in Hz.
    ear: The ear to play the tone in ('left' or 'right').
  """
  # Create a centered response button for use if the subject hears the tone.
  _, col2, _ = st.columns(3)
  with col2:
    if st.button('I hear the tone', key='pta_response',
                 icon=':material/hearing:'):
      # Calculate the response time.
      if st.session_state.pta_tone_start_time is not None:
        response_time_s = time.time() - st.session_state.pta_tone_start_time
        st.session_state.pta_tone_start_time = None  # Reset the start time.
      else:
        response_time_s = 0
      if st.session_state.pta_in_wait_period:
        # False positive response - add to logging, but don't do anything else.
        print('FALSE POSITIVE')
        st.session_state.pta_results.append(
          (ear, frequency_hz, st.session_state.pta_current_db_hl,
           'False positive', response_time_s))
      else:
        # Correct response - add to the results and decrease the level.
        st.session_state.pta_results.append(
          (ear, frequency_hz, st.session_state.pta_current_db_hl, True,
           response_time_s)
        )
        # Decrease the level by 10 dB, unless we hit the minimum level.
        st.session_state.pta_current_db_hl = max(
          st.session_state.pta_current_db_hl - 10, MIN_LEVEL_DB_HL
        )
        st.rerun()
  # The loop below will break in two scenarios that trigger a rerun:
  # 1. The subject presses the button above, indicating they heard the tone.
  # 2. The volume level reaches the maximum with no button press.
  while True:
    dbspl = calibration.dbhl_to_dbspl(st.session_state.pta_current_db_hl,
                                      frequency_hz,
                                      'PixelBuds_HughsonWestlake')
    dbspl += calibration.get_device_offset(
        frequency_hz, st.session_state.pta_device)
    amplitude = calibration.dbspl_to_amp(dbspl)
    print(f'\nVol: {st.session_state.pta_current_db_hl} dB HL, {dbspl} dB SPL')
    print(f'Current frequency: {frequency_hz} Hz for {ear} ear(s)')
    print(f'Amplitude: {amplitude}')
    random_onset_time = np.random.uniform(MIN_ONSET_TIME_S, MAX_ONSET_TIME_S)
    st.session_state.pta_in_wait_period = True
    time.sleep(random_onset_time)
    st.session_state.pta_in_wait_period = False
    if amplitude > 1:
      print('Warning: tone amplitude is greater than 1 - clipping.')
      amplitude = 1
    st.session_state.pta_tone_start_time = time.time()
    play_pulsed_tone(frequency_hz, amplitude, ear)
    time.sleep(RESPONSE_WINDOW_S)  # Give the user time to respond.
    st.session_state.pta_results.append(
      (ear, frequency_hz, st.session_state.pta_current_db_hl, False,
       RESPONSE_WINDOW_S)  # Use full response window for response time.
    )
    print('No response from user')
    # Increase the level by 5 dB, unless we are already at the max level.
    if st.session_state.pta_current_db_hl >= MAX_LEVEL_DB_HL:
      st.rerun()
    else:
      st.session_state.pta_current_db_hl = (
          st.session_state.pta_current_db_hl + 5)
      print(f'Increased volume to {st.session_state.pta_current_db_hl} db HL')

def get_start_level(
    freq: float,
    thresholds: dict,
    starting_offset: float=DEFAULT_STARTING_OFFSET_DBHL) -> float:
  """Returns the starting level for a tone, based on observed thresholds.

  The starting level is that which is starting_offset dB above the nearest
  measured threshold, where nearest is defined as the closest frequency to the
  one being tested.

  Args:
    freq: The frequency of the tone in Hz.
    thresholds: A dictionary where keys are frequencies in Hz and values are the
      corresponding hearing thresholds in dBHL.

  Returns:
    The starting level for the tone in dBHL.

  Raises:
    ValueError: If no thresholds have been measured.
  """
  if not thresholds:
    raise ValueError('No thresholds have been measured yet.')
  # Find the closest frequency to the one being tested.
  closest_freq = min(thresholds, key=lambda x: abs(x - freq))
  threshold = thresholds[closest_freq]
  # Set the starting level to 10 dB above the threshold.
  start_level = threshold + starting_offset
  return start_level

def generate_audiogram(thresholds_left: dict[float, float],
                       thresholds_right: dict[float, float],
                       thresholds_both: dict[float, float] = None
                       ) -> tuple[plt.Figure, list, list]:
  """Generates an audiogram plot from the hearing thresholds.

  Args:
    thresholds_left: A dictionary where keys are frequencies (in Hz) and
      values are the corresponding hearing thresholds (dBHL) for the left ear.
    thresholds_right: A dictionary where keys are frequencies (in Hz) and
      values are the corresponding hearing thresholds (dBHL) for the right ear.
    thresholds_both: A dictionary where keys are frequencies (in Hz) and
      values are the corresponding hearing thresholds (dBHL) for both ears
      combined.

  Returns:
    A tuple containing:
    1. A Matplotlib figure containing the audiogram plot
    2. Left ear audiogram as a list of (frequency, threshold) pairs
    3. Right ear audiogram as a list of (frequency, threshold) pairs
  """
  if thresholds_both:
    thresholds_left = thresholds_both
    thresholds_right = thresholds_both
  freqs = sorted(thresholds_left.keys())
  sorted_db_hl_values_left = []
  sorted_db_hl_values_right = []
  for freq in freqs:  # Iterate through frequencies in sorted order.
    sorted_db_hl_values_left.append(thresholds_left[freq])
    sorted_db_hl_values_right.append(thresholds_right[freq])
  fig, ax = plt.subplots()
  ax.semilogx(freqs, sorted_db_hl_values_left, marker='x',
              markeredgewidth=2, linestyle='-', color='#4285F4')
  ax.semilogx(freqs, sorted_db_hl_values_right, marker='o',
              linestyle='-', color='#DB4437')
  ax.legend(['Left ear', 'Right ear'])
  ax.set_xlabel('Frequency (kHz)')
  ax.set_ylabel('dB HL')
  ax.set_title('Audiogram')
  ax.set_xlim(125, 10000)
  ax.set_xticks(freqs)
  ax.set_xticklabels(
    [f'{f / 1000:.0f}' if f >= 1000 else f'{f / 1000:.2f}' for f in freqs])
  ax.set_ylim(85, -15)  # Flip the y-axis to match the audiogram convention.
  ax.set_yticks(np.arange(-10, 90, 10))
  ax.grid(True)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  left_audiogram = list(zip(freqs, sorted_db_hl_values_left))
  right_audiogram = list(zip(freqs, sorted_db_hl_values_right))
  return fig, left_audiogram, right_audiogram

def canned_data_button():
  """Displays the button to load canned data."""
  if st.button('Skip & load example', key='pta_canned_data',
               icon=':material/input:'):
    set_initial_demo_state()   # Reset to wipe any data already present.
    st.session_state.pta_state = 'Completed'
    # Load the canned data, ignoring the header lines delimited by '#'.
    df = pd.read_csv('assets/canned_pta_responses.csv', comment='#')
    st.session_state.pta_results = df.values.tolist()
    st.session_state.pta_using_canned_data = True
    print('Loaded canned data')
    # Calculate the thresholds from the canned data.
    for ear in ['left', 'right']:
      for freq in FREQS_HZ:
        ear_freq_results = [r[2:] for r in st.session_state.pta_results
                             if r[0] == ear and r[1] == freq]
        threshold = audiogram.get_threshold(ear_freq_results,
                                            min_level=MIN_LEVEL_DB_HL,
                                            max_level=MAX_LEVEL_DB_HL)
        if ear == 'left':
          st.session_state.pta_thresholds_left[freq] = threshold
        else:
          st.session_state.pta_thresholds_right[freq] = threshold
    st.rerun()

def _update_progress_bars():
  """Updates the progress bars based on the current state."""
  # Calculate total steps per ear.
  total_steps_per_ear = MAX_TONES_ADAPTIVE

  # Calculate current progress for each ear.
  # Filter for valid trials (True/False responses, not strings like
  # 'False positive').
  valid_trials = [r for r in st.session_state.pta_results
                  if isinstance(r[3], bool)]

  if st.session_state.pta_merge_lr:
    # If merged, both ears progress together.
    current_step = len(valid_trials)
    progress = min(1.0, current_step / total_steps_per_ear)
    if 'pta_progress_bar_left' in st.session_state:
      st.session_state.pta_progress_bar_left.progress(progress)
    if 'pta_progress_bar_right' in st.session_state:
      st.session_state.pta_progress_bar_right.progress(progress)
  else:
    # If separate, calculate progress for each ear.
    left_trials = [r for r in valid_trials if r[0] == 'left']
    right_trials = [r for r in valid_trials if r[0] == 'right']

    left_progress = min(1.0, len(left_trials) / total_steps_per_ear)
    right_progress = min(1.0, len(right_trials) / total_steps_per_ear)

    if 'pta_progress_bar_left' in st.session_state:
      st.session_state.pta_progress_bar_left.progress(left_progress)
    if 'pta_progress_bar_right' in st.session_state:
      st.session_state.pta_progress_bar_right.progress(right_progress)

def create_progress_bar():
  """Creates a progress bar for the Adaptive PTA test."""
  # Only show progress bar for the Adaptive PTA method, because the
  # Hughson-Westlake method is not deterministic and is hard to predict the
  # total number of steps.
  if st.session_state.pta_method != ADVANCED_METHOD_NAME:
    return
  # Set up the progress bars. Deliberately keep them small, so they don't act
  # as a cue for the user as each tone is played out.
  col_left, _ = st.columns([2, 16])
  with col_left:
    st.write('Left ear')
    st.session_state.pta_progress_bar_left = st.progress(0.0)
  col_right, _ = st.columns([2, 16])
  with col_right:
    st.write('Right ear')
    st.session_state.pta_progress_bar_right = st.progress(0.0)

  st.write('\n')
  # Initial update.
  _update_progress_bars()

def main_button_layout():
  """Displays the layout for the main buttons."""
  col1, col2, col3 = st.columns([1, 1, 1])
  with col1:
    start_button()
  with col2:
    cancel_button()
    st.write('')  # Add some whitespace.
  with col3:
    canned_data_button()

def display_settings():
  """Displays the settings for the PTA test."""
  st.subheader(common.SETTINGS_TITLE)
  st.write(common.SETTINGS_STRING)
  # Disable all settings once test has started.
  is_nal = st.session_state.get('app_target_audience') == 'NAL'
  settings_disabled = (st.session_state.pta_state == 'Running' or
                      st.session_state.pta_state == 'Completed' or
                      is_nal)
  st.toggle('Merge L/R', key='toggle_pta_merge_lr',
            help=common.MERGE_LR_HELP,
            disabled=settings_disabled)
  if st.session_state.toggle_pta_merge_lr:
    st.session_state.pta_merge_lr = True
    st.session_state.pta_current_ear = 'both'
  else:
    st.session_state.pta_merge_lr = False
    st.session_state.pta_current_ear = st.session_state.pta_current_ear
  # Add radio buttons to choose the headphone device (hidden for NAL).
  if not is_nal:
    device = st.radio('Headphone device:',
                      options=common.SUPPORTED_DEVICES,
                      index=0,
                      disabled=settings_disabled)
    st.session_state.pta_device = device
    if device == common.DEVICE_AIRPODS_PRO2:
      st.caption('Ensure all Hearing Assistance features are disabled.')
  else:
    st.session_state.pta_device = common.DEVICE_PIXEL_BUDS
  # Add a radio button to choose the test type.
  st.write('Select the test type:')
  test_type = st.radio('Test type:',
                        options=[BASIC_METHOD_NAME, ADVANCED_METHOD_NAME],
                        index=1,
                        disabled=settings_disabled)
  st.session_state.pta_method = test_type


def run_hughson_westlake():
  """Controls the top-level state of the Hughson-Westlake test."""
  current_frequency = FREQS_HZ[st.session_state.pta_current_freq_ind]
  res_current_freq = [r[2:4] for r in st.session_state.pta_results if
                      (r[0] == st.session_state.pta_current_ear and
                       r[1] == current_frequency)]

  print(res_current_freq)
  threshold = audiogram.get_threshold(res_current_freq,
                                      min_level=MIN_LEVEL_DB_HL,
                                      max_level=MAX_LEVEL_DB_HL)
  print(threshold)
  # Add the threshold to the dictionary for the current ear.
  if st.session_state.pta_current_ear == 'left':
    st.session_state.pta_thresholds_left[current_frequency] = threshold
  elif st.session_state.pta_current_ear == 'right':
    st.session_state.pta_thresholds_right[current_frequency] = threshold
  else:
    st.session_state.pta_thresholds_both[current_frequency] = threshold
  if threshold is not None:
    # We are done with that frequency and ear combination, so move on:
    # Finish the entire test, move to the right ear, or next frequency.
    if (st.session_state.pta_current_freq_ind == len(FREQS_HZ) - 1 and
        st.session_state.pta_current_ear in ['right', 'both']):
      # We are all done with the test for both ears.
      st.session_state.pta_state = 'Completed'
      st.session_state.pta_duration_s = (time.time() -
                                         st.session_state.pta_start_time)
      print('TEST COMPLETED')
      st.rerun()
    elif (st.session_state.pta_current_freq_ind == len(FREQS_HZ) - 1 and
          st.session_state.pta_current_ear == 'left'):
      # Were doing left ear, but it's done, so can move to the right ear.
      st.session_state.pta_current_ear = 'right'
      st.session_state.pta_current_freq_ind = 0
      st.session_state.pta_current_db_hl = START_LEVEL_DB_HL
      st.rerun()
    else:
      # Same ear, but move to next frequency.
      st.session_state.pta_current_freq_ind += 1
      current_frequency = FREQS_HZ[st.session_state.pta_current_freq_ind]
      thresholds = (st.session_state.pta_thresholds_left if
                    st.session_state.pta_current_ear == 'left' else
                    st.session_state.pta_thresholds_right if
                    st.session_state.pta_current_ear == 'right' else
                    st.session_state.pta_thresholds_both)
      initial_level = get_start_level(current_frequency, thresholds)
      if initial_level > MAX_LEVEL_DB_HL:
        initial_level = MAX_LEVEL_DB_HL
      st.session_state.pta_current_db_hl = initial_level
  # Run the next step of the Hughson-Westlake procedure.
  hughson_westlake_next_step(current_frequency,
                             st.session_state.pta_current_ear)

def adaptive_pta_next_step(ear):
  """Runs the next step of the adaptive PTA procedure.

  The adaptive PTA procedure is a modified version of the Hughson-Westlake
  procedure that uses a more complex algorithm to determine the next step.

  Args:
    ear: The ear to play the tone in ('left', 'right', 'both').

  Returns:
    The audiogram for the current ear.
  """
  print('Entering adaptive PTA next step')
  # Instantiate the selector, which defines the algorithm used for tone
  # selection.
  selector = pta_algorithms.HybridSelector()

  # Create a centered response button for use if the subject hears the tone.
  _, col2, _ = st.columns(3)
  with col2:
    if st.button('I hear the tone', key='pta_response',
                 icon=':material/hearing:'):
      # Calculate the response time.
      if st.session_state.pta_tone_start_time is not None:
        response_time_s = time.time() - st.session_state.pta_tone_start_time
        st.session_state.pta_tone_start_time = None  # Reset the start time.
      else:
        response_time_s = 0
      if st.session_state.pta_in_wait_period:
        # False positive response - add to logging, but don't do anything else.
        print('FALSE POSITIVE')
        st.session_state.pta_results.append(
          (ear, st.session_state.pta_current_freq,
           st.session_state.pta_current_db_hl, 'False positive',
           response_time_s))
      else:
        # Correct response.
        st.session_state.pta_results.append(
          (ear, st.session_state.pta_current_freq,
           st.session_state.pta_current_db_hl, True, response_time_s))
      st.rerun()

  print('Entering while loop')
  # The loop below will break in two scenarios that trigger a rerun:
  # 1. The subject presses the button above, indicating they heard the tone.
  # 2. The total number of tones played reaches the maximum number of tones.
  while True:
    # First, get the next tone based on results so far.
    # Filter results for the current ear and format for adaptive function
    # Expected format: list[tuple[float, float, bool]] (freq, level, response)
    # Only keep boolean responses for this ear, not false positives.
    past_results_formatted = [
        (res[1], res[2], bool(res[3])) # freq, level, boolean response
        for res in st.session_state.pta_results
        if res[0] == ear and isinstance(res[3], bool)
    ]
    if len(past_results_formatted) >= MAX_TONES_ADAPTIVE:
      # Test complete. Reconstruct final audiogram.
      print('Test complete for this ear. Reconstructing audiogram...')
      reconstructor = pta_algorithms.HybridLogisticReconstructor()
      # Note: reconstruct returns a dict of audiograms (Hybrid, Global, etc.)
      all_audiograms = reconstructor.reconstruct(past_results_formatted,
                                                 verbosity=1)
      # We use the 'Hybrid' result for the final output
      thresholds = all_audiograms['Hybrid']['thresholds']
      return thresholds

    # Get next stimulus
    stimulus = selector.next_stimulus(
      history=past_results_formatted,
      verbosity=1
    )
    print(stimulus)
    frequency_hz, dbhl = stimulus
    # Save these to the session state in case a button is pressed.
    st.session_state.pta_current_db_hl = dbhl
    st.session_state.pta_current_freq = frequency_hz
    dbspl = calibration.dbhl_to_dbspl(
        dbhl, frequency_hz, 'PixelBuds_Adaptive')
    dbspl += calibration.get_device_offset(
        frequency_hz, st.session_state.pta_device)
    amplitude = calibration.dbspl_to_amp(dbspl)
    print(f'\nVol: {dbhl} dB HL, {dbspl} dB SPL')
    print(f'Current frequency: {frequency_hz} Hz for {ear} ear(s)')
    print(f'Amplitude: {amplitude}')
    random_onset_time = np.random.uniform(MIN_ONSET_TIME_S, MAX_ONSET_TIME_S)
    st.session_state.pta_in_wait_period = True
    time.sleep(random_onset_time)
    st.session_state.pta_in_wait_period = False
    if amplitude > 1:
      print('Warning: tone amplitude is greater than 1 - clipping.')
      amplitude = 1
    st.session_state.pta_tone_start_time = time.time()
    play_pulsed_tone(frequency_hz, amplitude, ear)
    time.sleep(RESPONSE_WINDOW_S)  # Give the user time to respond.
    st.session_state.pta_results.append(
      (ear, frequency_hz, dbhl, False,
       RESPONSE_WINDOW_S)  # Use full response window for response time.
    )
    print('No response from user')
    _update_progress_bars()

def run_adaptive_pta():
  """Controls the top-level state of the adaptive PTA test."""
  test_is_complete = False
  if (st.session_state.pta_current_ear == 'both' and
      st.session_state.pta_thresholds_both):
    test_is_complete = True
  if (st.session_state.pta_current_ear == 'right' and
      st.session_state.pta_thresholds_right):
    test_is_complete = True
  if (st.session_state.pta_current_ear == 'left' and
      st.session_state.pta_thresholds_left):
    # Not done, but time to switch ears.
    st.session_state.pta_current_ear = 'right'
  if test_is_complete:
    st.session_state.pta_state = 'Completed'
    st.session_state.pta_duration_s = (time.time() -
                                       st.session_state.pta_start_time)
    print('TEST COMPLETED')
    st.rerun()
  else:
    # Not completed, so run the next step of the adaptive PTA procedure.
    threshold = adaptive_pta_next_step(st.session_state.pta_current_ear)
    print('DONE!!')
    print('threshold:', threshold)
    if st.session_state.pta_current_ear == 'left':
      st.session_state.pta_thresholds_left = threshold
    elif st.session_state.pta_current_ear == 'right':
      st.session_state.pta_thresholds_right = threshold
    else:
      st.session_state.pta_thresholds_both = threshold
    st.rerun()

def create_main_demo():
  """Controls the states and state transitions of the main demo."""
  if 'pta_initial_state_set' not in st.session_state:
    set_initial_demo_state()
  # Display material that is present for all states.
  create_intro_text()
  display_settings()
  common.display_preparation()
  # Hide the 'running man' icon that appears when the test is running.
  hide_streamlit_style = """
                  <style>
                  div[data-testid="stToolbar"] {
                  visibility: hidden;
                  height: 0%;
                  position: fixed;
                  }
                  """
  st.markdown(hide_streamlit_style, unsafe_allow_html=True)
  demo_button()
  st.write('')  # Add some whitespace.
  st.subheader('Take the test')
  main_button_layout()
  create_progress_bar()
  st.session_state.pta_in_wait_period = False  # Reset the wait period flag.
  # Take actions specific to the current state.
  if st.session_state.pta_state == 'Running':
    if st.session_state.pta_method == BASIC_METHOD_NAME:
      run_hughson_westlake()
    elif st.session_state.pta_method == ADVANCED_METHOD_NAME:
      run_adaptive_pta()
    else:
      raise ValueError('Invalid PTA method specified.')
  elif st.session_state.pta_state == 'Completed':
    st.write('')  # Add some whitespace.
    st.subheader('Results')
    # Display test duration.
    if st.session_state.pta_duration_s is not None:
      duration_m = int(st.session_state.pta_duration_s // 60)
      duration_s = int(st.session_state.pta_duration_s % 60)
      st.info(f'Test duration: {duration_m} min {duration_s} s')

    if st.session_state.pta_using_canned_data:
      st.write('The example data here indicate mild-to-moderate hearing loss '
               'in the higher frequencies. This is particularly pronounced for '
               'the left ear at 6 kHz and 8 kHz.')
    else:
      st.write('Here is an audiogram generated from the data collected during '
               'the test:')

    # Generate audiogram plot and data.
    audiogram_plot, left_audiogram, right_audiogram = generate_audiogram(
      st.session_state.pta_thresholds_left,
      st.session_state.pta_thresholds_right,
      st.session_state.pta_thresholds_both)
    st.pyplot(audiogram_plot)

    # Prepare data for download/email.
    if not st.session_state.pta_using_canned_data:
      full_csv = pta_results.generate_pta_full_results_csv(
          st.session_state.pta_results,
          st.session_state.pta_duration_s,
          st.session_state.pta_method
      )
      left_csv = pta_results.generate_audiogram_csv('Left', left_audiogram)
      right_csv = pta_results.generate_audiogram_csv('Right', right_audiogram)
      # Define files for zip.
      files_for_zip = [
          ('pta_full_results.csv', full_csv),
          ('pta_left_audiogram.csv', left_csv),
          ('pta_right_audiogram.csv', right_csv),
          ('pta_audiogram.png', audiogram_plot)
      ]
      zip_prefix = 'pta_results'
      test_name = 'Pure-Tone Audiometry'

      # Generate zip data and filename.
      zip_data = common.generate_zip_bytes(files_for_zip)
      timestamp = datetime.now().strftime('%Y%m%d_%H%M')
      zip_filename = f'UTC{timestamp}_{zip_prefix}.zip'

      # Save local backup if applicable; set flag to prevent multiple backups.
      if (st.session_state.is_running_locally and
          st.session_state.app_target_audience == 'NAL' and
          not st.session_state.get('pta_backup_saved', False)):
        common.save_local_backup(zip_data, zip_filename)
        st.session_state.pta_backup_saved = True
        print('PTA local backup saved.')

      st.write('\n\n')  # Add some space before the download button.
      st.download_button(
          label='Download results',
          data=zip_data,
          file_name=zip_filename,
          mime='application/zip',
          key='pta_manual_download'
      )
      # Display email form.
      common.display_email_results_form(test_name, files_for_zip, zip_prefix)

    # Close the figure after potential use in zip/email.
    plt.close(audiogram_plot)

    # Conditionally display interpretation.
    if st.session_state.app_target_audience != 'NAL':
      pta_results.display_interpretation()
    elif st.session_state.pta_using_canned_data:
      pta_results.display_interpretation()
