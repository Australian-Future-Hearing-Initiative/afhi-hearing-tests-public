"""Functionality for the 'pip' PTA demo."""

import numpy as np
import time
import pandas as pd
from datetime import datetime

import streamlit as st
import tempfile
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

import audio_tools
import calibration
import common
import pip_results
import pta_algorithms

FS_HZ = 44100  # Sampling frequency in Hz used to synthesize the tones.
N_PULSES = 3  # Number of pulses in the tone train.
PULSE_DURATION_S = 0.2  # Duration of each pulse in seconds.
GAP_DURATION_S = 0.25  # Duration of the gap between pulses in seconds.
DEMO_TONE_FREQ_HZ = 500  # Frequency of the demo tone in Hz.
DEMO_LEVEL_DB_HL = 65  # Level of the demo tone in dB HL.
CACHE_BUSTING_NOISE_DB_SPL = 0.1  # Level of noise to add to prevent caching.
SPINNER_DELAY_S = 1.5  # Keep the spinner spinning for this long.
DEFAULT_DATATYPE = 'float32'


def set_initial_demo_state():
  """Sets initial session state variables specific to the PIP test."""
  st.session_state.pip_state = 'Initial'  # 'Initial', 'Running', 'Completed'.
  st.session_state.pip_merge_lr = False  # Default value for binaural settings.
  if st.session_state.pip_merge_lr:
    st.session_state.pip_current_ear = 'both'
  else:
    st.session_state.pip_current_ear = common.DEFAULT_INITIAL_EAR
  st.session_state.pip_current_freq = None # Stores the current frequency.
  st.session_state.pip_current_db_hl = common.PTA_START_LEVEL_DB_HL
  st.session_state.pip_results = []  # List for storing results.
  st.session_state.pip_thresholds_left = {}  # Dict to store left thresholds.
  st.session_state.pip_thresholds_right = {}  # Dict to store right thresholds.
  st.session_state.pip_thresholds_both = {}  # Dict to store merged thresholds.
  st.session_state.pip_using_canned_data = False  # Indicate canned data loaded.
  st.session_state.pip_start_time = None # Start time of the test.
  st.session_state.pip_duration_s = None  # Duration of the test in seconds.
  st.session_state.pip_tone_start_time = None  # Start time of the current tone.
  st.session_state.pip_initial_state_set = True
  st.session_state.pip_backup_saved = False # Flag for local backup status.

def create_intro_text():
  """Creates the introductory text for the pure-tone audiometry demo."""
  st.title('Pip PTA Test')
  st.write('This is an experimental variation on pure-tone audiometry. The '
           'test requires the user to identify the number of pips they hear in '
           'a tone train.')

def play_pulsed_tone(frequency_hz: float,
                     amplitude: float,
                     ear: str = 'both',
                     n_pulses: int = N_PULSES,
                     pulse_duration_s: float = PULSE_DURATION_S,
                     gap_duration_s: float = GAP_DURATION_S,
                     datatype: str = DEFAULT_DATATYPE,
                     dither_db_spl: float = None):
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
  """
  if amplitude < 0 or amplitude > 1:  # Ensure the volume is between 0 and 1.
    raise ValueError('Volume must be between 0 and 1.')
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
  with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
    wavfile.write(temp_file.name, FS_HZ, tone)
    common.autoplay_audio(temp_file.name)  # Play the temporary file.

def demo_button():
  """Displays the demo button and supporting info."""
  st.write('Use the buttons below to hear examples of the tones that you will '
           'be asked to identify in this test.')
  col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
  dbspl = calibration.dbhl_to_dbspl(DEMO_LEVEL_DB_HL,
                                    DEMO_TONE_FREQ_HZ,
                                    'PixelBuds_Pip')
  dbspl += calibration.get_device_offset(
      DEMO_TONE_FREQ_HZ, st.session_state.pip_device)
  tone_amp = calibration.dbspl_to_amp(dbspl)
  with col1:
    if st.button('Silence', icon=':material/play_arrow:'):
      play_pulsed_tone(DEMO_TONE_FREQ_HZ, 0, n_pulses=1)
  with col2:
    if st.button('One pip', icon=':material/play_arrow:'):
      play_pulsed_tone(DEMO_TONE_FREQ_HZ, tone_amp, n_pulses=1)
  with col3:
    if st.button('Two pips', icon=':material/play_arrow:'):
      play_pulsed_tone(DEMO_TONE_FREQ_HZ, tone_amp, n_pulses=2)
  with col4:
    if st.button('Three pips', icon=':material/play_arrow:'):
      play_pulsed_tone(DEMO_TONE_FREQ_HZ, tone_amp, n_pulses=3)

def start_button():
  """Displays the start button and handles its functionality."""
  if st.button('Start the test', key='pta_start_test',
               icon=':material/play_arrow:',
               disabled=st.session_state.pip_state == 'Running'):
    set_initial_demo_state()
    st.session_state.pip_state = 'Running'
    st.session_state.pip_start_time = time.time()
    st.rerun()

def cancel_button():
  """Displays the cancel button and handles its functionality."""
  if st.button('Cancel the test', key='pta_cancel_test',
               icon=':material/cancel:',
               disabled=st.session_state.pip_state != 'Running'):
    set_initial_demo_state()
    st.rerun()

def user_response(frequency_hz: float, ear: str, n_pips: int):
  """Handles app flow when the user responds to the test.

  Args:
    n_pips: The number of pips the user indicates they heard.
  """
  print(f'User response: {n_pips} pips')
  actual_n_pips = st.session_state.pip_latest_n_pips
  # Calculate the response time
  if st.session_state.pip_tone_start_time is not None:
    response_time_s = time.time() - st.session_state.pip_tone_start_time
    st.session_state.pip_tone_start_time = None  # Reset the start time
  else:
    response_time_s = 0
  if n_pips == actual_n_pips:
    print('Correct response')
    result = True
  else:
    print('Incorrect response')
    result = False

  st.session_state.pip_results.append(
    (ear, frequency_hz, st.session_state.pip_current_db_hl, result,
     response_time_s)
  )
  # Rerun the app to get the next tone.
  st.rerun()

def pip_test_next_step(frequency_hz, ear):
  """Runs the next step of the Pip test.

  Args:
    frequency_hz: The frequency of the tone in Hz.
    ear: The ear to play the tone in ('left' or 'right').
  """
  # Create the four buttons for the four different choices. In reality, there
  # are only three types of tones (1, 2 or 3 pips), but the user might not hear
  # the tone, in which case they should press the 'No pips' button.
  st.write('')
  st.write('Press the button that best matches what you heard:')
  st.write('')
  # Creates buttons for tones.
  col1, col2, col3, col4, _ = st.columns([2, 1, 1, 1, 2])
  with col1:
    spinner_placeholder = st.empty()
  with col2:
    if st.button('1 pip', key='pip_1_pips'):
      user_response(frequency_hz, ear, 1)
    if st.button('0 pips / Not sure / Silence', key='pip_0_pips'):
      user_response(frequency_hz, ear, 0)
  with col3:
    if st.button('2 pips', key='pip_2_pips'):
      user_response(frequency_hz, ear, 2)

  with col4:
    if st.button('3 pips', key='pip_3_pips'):
      user_response(frequency_hz, ear, 3)

  # Randomly select a number of pips from the set {1, 2, 3}.
  n_pips = np.random.choice([1, 2, 3])
  st.session_state.pip_latest_n_pips = n_pips
  # Play the tone.
  dbspl = calibration.dbhl_to_dbspl(st.session_state.pip_current_db_hl,
                                    frequency_hz,
                                    'PixelBuds_Pip')
  dbspl += calibration.get_device_offset(
      frequency_hz, st.session_state.pip_device)
  amplitude = calibration.dbspl_to_amp(dbspl)
  if amplitude > 1:
    amplitude = 1
    print('Warning - Amplitude clipped to 1')
  print(f'\nVol: {st.session_state.pip_current_db_hl} dB HL, {dbspl} dB SPL')
  print(f'Current frequency: {frequency_hz} Hz for {ear} ear(s)')
  print(f'Amplitude: {amplitude}')
  print(f'Generated a {n_pips}-pip tone')
  with spinner_placeholder, st.spinner('Playing ...'):
    st.session_state.pip_tone_start_time = time.time()
    play_pulsed_tone(frequency_hz, amplitude, ear, n_pulses=n_pips)
    time.sleep(SPINNER_DELAY_S)


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
    1. A Matplotlib figure containing the audiogram plot.
    2. Left ear audiogram as a list of (frequency, threshold) pairs.
    3. Right ear audiogram as a list of (frequency, threshold) pairs.
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
    st.session_state.pip_state = 'Completed'
    # Load the canned data, ignoring the header lines delimited by '#'.
    df = pd.read_csv('assets/canned_pta_responses.csv', comment='#')
    st.session_state.pip_results = df.values.tolist()
    st.session_state.pip_using_canned_data = True
    print('Loaded canned data')
    # Calculate the thresholds from the canned data.
    reconstructor = pta_algorithms.HybridLogisticReconstructor()
    for ear in ['left', 'right']:
      # Filter results for the current ear.
      ear_results = [
          (r[1], r[2], bool(r[3])) for r in st.session_state.pip_results
          if r[0] == ear
      ]
      if ear_results:
        all_audiograms = reconstructor.reconstruct(ear_results, verbosity=1)
        thresholds = all_audiograms['Hybrid']['thresholds']
        if ear == 'left':
          st.session_state.pip_thresholds_left = thresholds
        else:
          st.session_state.pip_thresholds_right = thresholds
    st.rerun()

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
  """Displays the settings for the PIP test."""
  st.subheader(common.SETTINGS_TITLE)
  st.write(common.SETTINGS_STRING)
  # Disable all settings once test has started
  is_nal = st.session_state.get('app_target_audience') == 'NAL'
  settings_disabled = (st.session_state.pip_state == 'Running' or
                      st.session_state.pip_state == 'Completed' or
                      is_nal)
  st.toggle('Merge L/R', key='toggle_pip_merge_lr',
            help=common.MERGE_LR_HELP,
            disabled=settings_disabled)
  if st.session_state.toggle_pip_merge_lr:
    st.session_state.pip_merge_lr = True
    st.session_state.pip_current_ear = 'both'
  else:
    st.session_state.pip_merge_lr = False
    st.session_state.pip_current_ear = st.session_state.pip_current_ear
  # Add radio buttons to choose the headphone device (hidden for NAL).
  if not is_nal:
    device = st.radio('Headphone device:',
                      options=common.SUPPORTED_DEVICES,
                      index=0,
                      disabled=settings_disabled)
    st.session_state.pip_device = device
    if device == common.DEVICE_AIRPODS_PRO2:
      st.caption('Ensure all Hearing Assistance features are disabled.')
  else:
    st.session_state.pip_device = common.DEVICE_PIXEL_BUDS

def _update_progress_bars():
  """Updates the progress bars based on the current state."""
  total_steps_per_ear = common.PTA_MAX_TRIALS_PER_EAR
  all_trials = st.session_state.pip_results

  if st.session_state.pip_merge_lr:
    current_step = len(all_trials)
    progress = min(1.0, current_step / total_steps_per_ear)
    if 'pip_progress_bar_left' in st.session_state:
      st.session_state.pip_progress_bar_left.progress(progress)
    if 'pip_progress_bar_right' in st.session_state:
      st.session_state.pip_progress_bar_right.progress(progress)
  else:
    left_trials = [r for r in all_trials if r[0] == 'left']
    right_trials = [r for r in all_trials if r[0] == 'right']
    left_progress = min(1.0, len(left_trials) / total_steps_per_ear)
    right_progress = min(1.0, len(right_trials) / total_steps_per_ear)
    if 'pip_progress_bar_left' in st.session_state:
      st.session_state.pip_progress_bar_left.progress(left_progress)
    if 'pip_progress_bar_right' in st.session_state:
      st.session_state.pip_progress_bar_right.progress(right_progress)

def create_progress_bar():
  """Creates a progress bar for the PIP test."""
  # Keep bars small so they don't act as a cue during tone playback.
  col_left, _ = st.columns([2, 16])
  with col_left:
    st.write('Left ear')
    st.session_state.pip_progress_bar_left = st.progress(0.0)
  col_right, _ = st.columns([2, 16])
  with col_right:
    st.write('Right ear')
    st.session_state.pip_progress_bar_right = st.progress(0.0)
  st.write('\n')
  # Initial update.
  _update_progress_bars()

def create_main_demo():
  """Controls the states and state transitions of the main demo."""
  if 'pip_initial_state_set' not in st.session_state:
    set_initial_demo_state()
  # Display material that is present for all states.
  create_intro_text()
  display_settings()
  common.display_preparation()
  demo_button()
  st.write('')  # Add some whitespace.
  st.subheader('Take the test')
  main_button_layout()
  # Take actions specific to the current state.
  if st.session_state.pip_state == 'Running':
    create_progress_bar()
    # Instantiate the selector.
    selector = pta_algorithms.HybridSelector()
    ear = st.session_state.pip_current_ear
    # Filter results for the current ear and format for adaptive function
    # Expected format: list[tuple[float, float, bool]] (freq, level, response)
    # Only keep boolean responses for this ear (ignoring any non-bool if any)
    past_results_formatted = [
        (res[1], res[2], bool(res[3])) # freq, level, boolean response
        for res in st.session_state.pip_results
        if res[0] == ear
    ]

    # Check for completion of the current ear
    if len(past_results_formatted) >= common.PTA_MAX_TRIALS_PER_EAR:
      # Test complete for this ear. Reconstruct audiogram.
      print('Test complete for this ear. Reconstructing audiogram...')
      reconstructor = pta_algorithms.HybridLogisticReconstructor()
      # Note: reconstruct returns a dict of audiograms (Hybrid, Global, etc.)
      all_audiograms = reconstructor.reconstruct(past_results_formatted,
                                                 verbosity=1)
      # We use the 'Hybrid' result for the final output.
      thresholds = all_audiograms['Hybrid']['thresholds']
      if ear == 'left':
        st.session_state.pip_thresholds_left = thresholds
      elif ear == 'right':
        st.session_state.pip_thresholds_right = thresholds
      else:
        st.session_state.pip_thresholds_both = thresholds
      # Move to next ear or complete.
      if ear == 'left':
        print('MOVED TO RIGHT EAR ---------')
        st.session_state.pip_current_ear = 'right'
        # Reset any necessary state for the new ear if needed,
        # but selector handles new history automatically.
        st.rerun()
      else:
        # Done with right or both
        st.session_state.pip_state = 'Completed'
        st.session_state.pip_duration_s = (time.time() -
                                           st.session_state.pip_start_time)
        print('TEST COMPLETED')
        st.rerun()

    else:
      # Get next stimulus
      stimulus = selector.next_stimulus(
        history=past_results_formatted,
        verbosity=1
      )
      frequency_hz, dbhl = stimulus
      st.session_state.pip_current_freq = frequency_hz
      st.session_state.pip_current_db_hl = dbhl
      # Run the presentation step.
      pip_test_next_step(frequency_hz, ear)
      _update_progress_bars()

  elif st.session_state.pip_state == 'Completed':
    st.write('')  # Add some whitespace.
    st.subheader('Results')
    # Display test duration.
    if st.session_state.pip_duration_s is not None:
      duration_m = int(st.session_state.pip_duration_s // 60)
      duration_s = int(st.session_state.pip_duration_s % 60)
      st.info(f'Test duration: {duration_m} min {duration_s} s')

    if st.session_state.pip_using_canned_data:
      st.write('The example data here indicate mild-to-moderate hearing loss '
               'in the higher frequencies. This is particularly pronounced for '
               'the left ear at 6 kHz and 8 kHz.')
    else:
      st.write('Here is an audiogram generated from the data collected during '
               'the test:')

    # Generate audiogram plot and data.
    audiogram_plot, left_audiogram, right_audiogram = generate_audiogram(
      st.session_state.pip_thresholds_left,
      st.session_state.pip_thresholds_right,
      st.session_state.pip_thresholds_both)
    st.pyplot(audiogram_plot)

    # Prepare data for download/email.
    if not st.session_state.pip_using_canned_data:
      full_csv = pip_results.generate_pip_full_results_csv(
          st.session_state.pip_results,
          st.session_state.pip_duration_s
      )
      left_csv = pip_results.generate_pip_audiogram_csv('Left', left_audiogram)
      right_csv = pip_results.generate_pip_audiogram_csv('Right',
                                                         right_audiogram)
      # Define files for zip.
      files_for_zip = [
          ('pip_full_results.csv', full_csv),
          ('pip_left_audiogram.csv', left_csv),
          ('pip_right_audiogram.csv', right_csv),
          ('pip_audiogram.png', audiogram_plot)
      ]
      zip_prefix = 'pip_results'
      test_name = 'Pip PTA Test'
      # Display download button.
      zip_data = common.generate_zip_bytes(files_for_zip)
      timestamp = datetime.now().strftime('%Y%m%d_%H%M')
      zip_filename = f'UTC{timestamp}_{zip_prefix}.zip'

      # Save local backup if applicable; set flag to prevent multiple backups.
      if (st.session_state.is_running_locally and
          st.session_state.app_target_audience == 'NAL' and
          not st.session_state.get('pip_backup_saved', False)):
        common.save_local_backup(zip_data, zip_filename)
        st.session_state.pip_backup_saved = True
        print('PIP local backup saved.')

      st.write('\n\n')  # Add some space before the download button.
      st.download_button(
          label='Download results',
          data=zip_data,
          file_name=zip_filename,
          mime='application/zip',
          key='pip_manual_download'
      )
      # Display email form.
      common.display_email_results_form(test_name, files_for_zip, zip_prefix)

    # Close the figure after potential use in zip/email.
    plt.close(audiogram_plot)
    # Conditionally display interpretation
    if st.session_state.app_target_audience != 'NAL':
      pip_results.display_interpretation()
    elif st.session_state.pip_using_canned_data: # Handle canned data case.
      pip_results.display_interpretation()


