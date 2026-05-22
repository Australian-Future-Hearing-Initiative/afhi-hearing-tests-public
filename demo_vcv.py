"""Functionality for the adaptive consonant confusion demo part of the app."""

import glob
from datetime import datetime
import math
import os
import random
import shutil
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd
from scipy.io import wavfile
import streamlit as st

import analytics
import audio_synthesis
import bayesian_vcv_estimator
import calibration
import common
import custom_vcv_loader
import vcv_results


CLEAN_WAV_DIR = os.path.join(common.PREFERRED_STIMULI_DIR, 'clean_standardised')
SYNTHETIC_WAV_DIR = os.path.join(common.PREFERRED_STIMULI_DIR, 'synthetic')
SNR_OPTIONS_DB = ('-12', '-9', '-6', '-3', '0', '+3', '+6', '+9', '+12', '+60')
DEFAULT_SNR_OPTIONS_DB = ('-12', '-6', '0', '+6')
DEFAULT_N_TEST_TRIALS = 90
NOISE_RAMP_DURATION_S = 0.1
N_PRACTICE_TRIALS = 5
PRACTICE_SNR_DB = 10.0
# Volume level labels mapped to dB SPL values.
# Hearing profile labels mapped to dB SPL values.
HEARING_PROFILES = {
    'Normal hearing': 63.0,
    'Mild hearing loss': 68.0,
    'Moderate hearing loss': 73.0,
}
_DB_TO_HEARING_PROFILE = {
    v: k for k, v in HEARING_PROFILES.items()
}

DEFAULTS = {
    # Settings.
    'vcv_test_mode': 'Adaptive',
    'vcv_stimuli_type': 'Human',
    'vcv_n_total_trials': DEFAULT_N_TEST_TRIALS,
    'vcv_merge_lr': False,
    'vcv_snr_levels': DEFAULT_SNR_OPTIONS_DB,
    'vcv_volume_db_spl': calibration.DEFAULT_VCV_DB_SPL,

    # State flags.
    'vcv_freeze_settings': False,
    'vcv_test_completed': False,
    'vcv_play_button_disabled': False,
    'vcv_backup_saved': False,
    'vcv_initial_state_set': False,
    'vcv_practice_completed': False,
    'vcv_practice_trials_count': 0,

    # Data containers (Must exist to prevent AttributeErrors).
    'vcv_responses': [],
    'vcv_is_practice_trial': False,
    'vcv_play_count': 0,
    'vcv_completed_stimuli_current_ear': 0,
    'vcv_pending_audio': None,
    'base_files_by_consonant': {},
    'vcv_estimators': None,
    'last_played_audio': None,
    'vcv_tone_start_time': None,
    'vcv_last_snr': None,
    'vcv_last_condition_key': None,
    'vcv_df': None,
    'vcv_final_estimates': None,
    'vcv_confusion_results': None,
    'vcv_results_left': None,
    'vcv_results_right': None,
    'vcv_practice_feedback': None,

    # NAL study mode: per-trial WAV saving.
    'vcv_wav_save_dir': None,
    'vcv_last_saved_wav_path': None,

    # Custom stimuli (ZIP upload).
    'vcv_custom_stimuli_loaded': False,
    'vcv_custom_wav_dir': None,
    'vcv_custom_consonants': [],
    'vcv_custom_load_errors': [],
}

DEFAULT_MERGE_LR = False
NOISE_TYPE_FOR_SYNTHESIS = 'Advanced Speech-Shaped Noise'

# Consonant data is defined in bayesian_vcv_estimator.
# Local aliases keep references concise throughout this module.
CONSONANT_LABELS = bayesian_vcv_estimator.CONSONANT_LABELS
CONSONANT_SNR_FLOOR_DB = (
    bayesian_vcv_estimator.CONSONANT_SNR_FLOOR_DB
)

# Consonant display order, derived from class definitions.
ORDERED_LABELS = bayesian_vcv_estimator.ORDERED_CONSONANTS
# Circular layout order (clockwise from top).
STANDARD_CIRCLE_ORDER = [
    'D', 'G', 'K', 'N', 'S', 'SH', 'T', 'V', 'Z', 'B'
]


def _get_active_consonant_set() -> dict[str, str]:
  """Returns the consonant labels dict for the current stimuli type."""
  if st.session_state.vcv_stimuli_type == 'Custom':
    return {c: bayesian_vcv_estimator.CONSONANT_LABELS[c]
            for c in st.session_state.vcv_custom_consonants
            if c in bayesian_vcv_estimator.CONSONANT_LABELS}
  # For Human/Synthetic modes, only standard 10 consonants have audio files.
  standard = {'B', 'D', 'G', 'K', 'N', 'S', 'SH', 'T', 'V', 'Z'}
  return {k: v for k, v in CONSONANT_LABELS.items() if k in standard}


def _get_active_circle_order() -> list[str]:
  """Returns the button ring order for the current consonant set."""
  if st.session_state.vcv_stimuli_type == 'Custom':
    active = set(st.session_state.vcv_custom_consonants)
    # Preserve the standard order for standard consonants, append any new ones.
    order = [c for c in STANDARD_CIRCLE_ORDER if c in active]
    extra = sorted(list(active - set(STANDARD_CIRCLE_ORDER)))
    return order + extra
  return STANDARD_CIRCLE_ORDER

def _get_all_base_names() -> dict[str, list[str]]:
  """Gets unique base filenames organized by consonant."""
  if st.session_state.vcv_stimuli_type == 'Custom':
    return _get_custom_base_names()

  if st.session_state.vcv_stimuli_type == 'Human':
    wav_dir = CLEAN_WAV_DIR
  else:
    wav_dir = SYNTHETIC_WAV_DIR

  if not os.path.isdir(wav_dir):
    st.error(f"Stimuli directory not found: {wav_dir}")
    return {}

  active_set = _get_active_consonant_set()
  all_files = glob.glob(os.path.join(wav_dir, '*.wav'))
  base_names_by_consonant = {c: [] for c in active_set}
  for f_path in all_files:
    base_name = os.path.basename(f_path)
    name_part = os.path.splitext(base_name)[0]
    consonant = get_correct_answer(name_part)
    if consonant in base_names_by_consonant:
      base_names_by_consonant[consonant].append(name_part)

  return base_names_by_consonant


def _get_custom_base_names() -> dict[str, list[str]]:
  """Builds base-name dict from custom stimuli on disk."""
  wav_dir = st.session_state.vcv_custom_wav_dir
  if not wav_dir or not os.path.isdir(wav_dir):
    return {}

  result = {}
  for consonant in st.session_state.vcv_custom_consonants:
    vcv_token = bayesian_vcv_estimator.CONSONANT_LABELS.get(consonant)
    if not vcv_token:
      continue

    # Look for the folder by VCV token name (case-insensitive).
    folder_path = None
    for entry in os.listdir(wav_dir):
      full = os.path.join(wav_dir, entry)
      if os.path.isdir(full) and entry.lower() == vcv_token:
        folder_path = full
        break

    # Also check for a wrapper folder one level down.
    if folder_path is None:
      for entry in os.listdir(wav_dir):
        sub = os.path.join(wav_dir, entry)
        if os.path.isdir(sub):
          for sub_entry in os.listdir(sub):
            full = os.path.join(sub, sub_entry)
            if os.path.isdir(full) and sub_entry.lower() == vcv_token:
              folder_path = full
              break
          if folder_path:
            break

    if folder_path is None:
      result[consonant] = []
      continue

    wav_files = [
        os.path.splitext(f)[0]
        for f in sorted(os.listdir(folder_path))
        if f.lower().endswith('.wav') and not f.startswith('.')
    ]
    result[consonant] = wav_files

  return result

def get_stimulus_for_consonant(consonant: str) -> str | None:
  """Selects a random WAV stimulus path for a given consonant."""
  available_files = st.session_state.base_files_by_consonant.get(consonant)
  if not available_files:
    return None
  selected_base_name = random.choice(available_files)

  if st.session_state.vcv_stimuli_type == 'Custom':
    return _get_custom_wav_path(consonant, selected_base_name)
  if st.session_state.vcv_stimuli_type == 'Human':
    wav_dir = CLEAN_WAV_DIR
  else:
    wav_dir = SYNTHETIC_WAV_DIR
  return os.path.join(wav_dir, f'{selected_base_name}.wav')


def _get_custom_wav_path(consonant: str, base_name: str) -> str | None:
  """Resolves a custom stimulus base name to a full WAV path."""
  wav_dir = st.session_state.vcv_custom_wav_dir
  if not wav_dir:
    return None

  vcv_token = bayesian_vcv_estimator.CONSONANT_LABELS.get(consonant)
  if not vcv_token:
    return None

  # Search for the folder matching this VCV token.
  for root, dirs, _ in os.walk(wav_dir):
    for d in dirs:
      if d.lower() == vcv_token:
        candidate = os.path.join(root, d, f'{base_name}.wav')
        if os.path.isfile(candidate):
          return candidate
  return None

def get_random_audio_file_for_practice() -> str | None:
  """Selects a random WAV stimulus name and constructs the full path."""
  active_set = _get_active_consonant_set()
  random_consonant = random.choice(list(active_set.keys()))
  return get_stimulus_for_consonant(random_consonant)

def initialize_estimators():
  """Initializes ZEST estimators with class-level priors."""
  estimators = {}
  ears = (
      ['both']
      if st.session_state.vcv_merge_lr
      else ['left', 'right']
  )
  active_set = _get_active_consonant_set()
  for ear in ears:
    for consonant in active_set:
      prior_mean = (
          bayesian_vcv_estimator
          .CONSONANT_INITIAL_SNR_DB[consonant]
      )
      estimators[(ear, consonant)] = (
          bayesian_vcv_estimator.ZestEstimator(
              prior_mean=prior_mean,
              prior_sd=bayesian_vcv_estimator.PRIOR_SD,
          )
      )
  st.session_state.vcv_estimators = estimators

def set_initial_demo_state():
  """Ensures all state keys exist."""
  for key, value in DEFAULTS.items():
    if key not in st.session_state:
      # Create a copy if the value is mutable to avoid shared state bugs
      if isinstance(value, list):
        st.session_state[key] = list(value)
      elif isinstance(value, dict):
        st.session_state[key] = dict(value)
      else:
        st.session_state[key] = value

  if not st.session_state.base_files_by_consonant:
    st.session_state.base_files_by_consonant = _get_all_base_names()

  if ('vcv_estimators' not in st.session_state or
      st.session_state.vcv_estimators is None):
    st.session_state.vcv_current_ear = (
        'both' if st.session_state.vcv_merge_lr else common.DEFAULT_INITIAL_EAR
    )
    initialize_estimators()

  st.session_state.vcv_initial_state_set = True

def reset_results_only():
  """Callback: Clears test progress but PRESERVES user settings."""
  st.session_state.vcv_responses = []
  st.session_state.vcv_play_count = 0
  st.session_state.vcv_completed_stimuli_current_ear = 0
  st.session_state.vcv_test_completed = False
  st.session_state.vcv_practice_completed = False
  st.session_state.vcv_practice_trials_count = 0
  st.session_state.vcv_is_practice_trial = False
  st.session_state.vcv_practice_feedback = None

  st.session_state.vcv_estimators = None
  st.session_state.base_files_by_consonant = {}

  # Clean up custom stimuli temp dir on reset.
  if st.session_state.get('vcv_custom_wav_dir'):
    custom_vcv_loader.cleanup_temp_dir(
        st.session_state.vcv_custom_wav_dir
    )
    st.session_state.vcv_custom_wav_dir = None
    st.session_state.vcv_custom_stimuli_loaded = False
    st.session_state.vcv_custom_consonants = []
    st.session_state.vcv_custom_load_errors = []

  set_initial_demo_state()

def create_intro_text():
  """Creates the introductory text for the consonant confusion test."""
  st.title('Consonant Confusion Test')
  st.write('Consonant confusion tests can provide a more detailed and nuanced '
           "understanding of an individual's hearing abilities than "
           'audiograms, which fail to capture the complexities of speech '
           'perception in challenging listening environments. By '
           'analysing patterns of consonant errors, we can pinpoint specific '
           'areas of difficulty. This can quantify hearing aid benefit, '
           'potentially leading to better hearing aid fitting.')
  st.write('This test will play a series of audio samples of consonants for '
           'you to identify.')

def get_correct_answer(audio_file_name):
  """Extracts the correct consonant from an audio file path.

  The lookup strategy depends on the stimuli type:
    - **Human / Synthetic**: the VCV token is embedded in the filename
      (e.g. ``VCV_aba_1_60SNR.wav``), so we match against the basename.
    - **Custom**: the VCV token is the parent folder name
      (e.g. ``ABA/speaker1.wav``); individual filenames are arbitrary
      and must not be used for identification.
  """
  # Check against the full label set (not just active) so that
  # file-based lookups always work regardless of mode.

  if st.session_state.get('vcv_stimuli_type') == 'Custom':
    # Custom stimuli: consonant is encoded in the parent directory.
    parent_dir = os.path.basename(
        os.path.dirname(audio_file_name)
    ).lower()
    for consonant, label in CONSONANT_LABELS.items():
      if label == parent_dir:
        return consonant
    return None

  # Human / Synthetic stimuli: VCV token is part of the filename.
  name_to_check = os.path.basename(audio_file_name)
  for consonant, label in CONSONANT_LABELS.items():
    if label in name_to_check:
      return consonant
  return None

def _bundle_wav_files():
  """Zips saved WAV files and writes the archive to local_results/."""
  wav_dir = st.session_state.vcv_wav_save_dir
  if not wav_dir or not os.path.isdir(wav_dir):
    return
  wav_files = sorted(glob.glob(os.path.join(wav_dir, '*.wav')))
  if not wav_files:
    return
  timestamp = datetime.now().strftime('%Y%m%d_%H%M')
  zip_filename = f'UTC{timestamp}_vcv_audio.zip'
  zip_path = os.path.join('local_results', zip_filename)
  os.makedirs('local_results', exist_ok=True)
  with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for wav_path in wav_files:
      zf.write(wav_path, os.path.basename(wav_path))
  print(f'VCV audio zip saved: {zip_path} ({len(wav_files)} files)')
  shutil.rmtree(wav_dir)
  print(f'Removed temporary WAV directory: {wav_dir}')


def complete_test():
  st.session_state.vcv_test_completed = True
  st.session_state.vcv_play_button_disabled = True
  active_set = _get_active_consonant_set()
  if st.session_state.vcv_test_mode == 'Adaptive':
    # Use the class-ordered labels, filtered to active set.
    all_possible_labels = [
        c for c in ORDERED_LABELS if c in active_set
    ]
  else:
    all_possible_labels = list(active_set.keys())

  if st.session_state.vcv_test_mode == 'Adaptive':
    final_columns = ['Ear', 'Filename', 'Target Consonant', 'Response',
                     'Is Correct', 'Response Time (s)', 'SNR (dB)']
  else:
    final_columns = ['Ear', 'Filename', 'Response', 'Correct Answer',
                     'Response Time (s)', 'SNR (dB)']

  df = pd.DataFrame(st.session_state.vcv_responses, columns=final_columns)
  df.index.name = 'Trial Number'
  df.index += 1
  st.session_state.vcv_df = df

  st.session_state.vcv_confusion_results = {}
  if st.session_state.vcv_merge_lr:
    ears_to_process = ['both']
  else:
    ears_to_process = ['left', 'right']

  for ear in ears_to_process:
    ear_data = df[df['Ear'] == ear]
    if st.session_state.vcv_test_mode == 'Adaptive':
      results_to_analyze = ear_data[
          ['Response', 'Target Consonant']
      ].values.tolist()
    else:
      results_to_analyze = ear_data[
          ['Response', 'Correct Answer']
      ].values.tolist()

    if results_to_analyze:
      analysis_result = analytics.analyze_results(
          results_to_analyze, all_possible_labels
      )
      st.session_state.vcv_confusion_results[ear] = analysis_result
    else:
      st.session_state.vcv_confusion_results[ear] = None

  if st.session_state.vcv_test_mode == 'Adaptive':
    final_estimates = []
    for (ear, consonant), estimator in st.session_state.vcv_estimators.items():
      srt, uncertainty = estimator.get_estimate()
      final_estimates.append({
          'Ear': ear, 'Consonant': consonant, 'SRT (dB)': srt,
          'Uncertainty (SD)': uncertainty, 'Trials': len(estimator.history)
      })
    st.session_state.vcv_final_estimates = pd.DataFrame(final_estimates)
  else:
    if st.session_state.vcv_merge_lr:
      st.session_state.vcv_results_left = (
          st.session_state.vcv_confusion_results.get('both')
      )
      st.session_state.vcv_results_right = (
          st.session_state.vcv_confusion_results.get('both')
      )
    else:
      st.session_state.vcv_results_left = (
          st.session_state.vcv_confusion_results.get('left')
      )
      st.session_state.vcv_results_right = (
          st.session_state.vcv_confusion_results.get('right')
      )
    if 'vcv_final_estimates' in st.session_state:
      st.session_state.vcv_final_estimates = None

  # Note: WAV files from vcv_wav_save_dir are included in the main results
  # zip by vcv_results.py, so no separate bundling is needed here.

  st.rerun()

def prepare_next_trial():
  """Schedules the next trial and sets up pending audio."""
  condition_key, selected_snr_db = schedule_next_trial(
      st.session_state.vcv_estimators
  )
  st.session_state.vcv_last_condition_key = condition_key
  st.session_state.vcv_last_snr = selected_snr_db

  st.session_state.vcv_pending_audio = {
      'consonant': condition_key[1],
      'snr': selected_snr_db,
      'ear': condition_key[0]
  }


def _rename_saved_wav(response_label: str):
  """Renames the last saved WAV to include the actual response."""
  old_path = st.session_state.vcv_last_saved_wav_path
  if not old_path or not os.path.exists(old_path):
    return
  new_path = old_path.replace('_response_PENDING_',
                              f'_response_{response_label}_')
  os.rename(old_path, new_path)
  st.session_state.vcv_last_saved_wav_path = None
  print(f'  Renamed WAV: {os.path.basename(new_path)}')


def handle_response_button_click(button_label):
  """Handles the logic when a response button is clicked."""
  if st.session_state.last_played_audio is None:
    return

  correct_answer = get_correct_answer(st.session_state.last_played_audio)

  if st.session_state.vcv_tone_start_time is not None:
    response_time_s = time.time() - st.session_state.vcv_tone_start_time
    st.session_state.vcv_tone_start_time = None
  else:
    response_time_s = 0

  if st.session_state.vcv_is_practice_trial:
    # Practice run logic: Don't save results.
    is_correct = button_label == correct_answer
    st.session_state.vcv_practice_feedback = {
        'is_correct': is_correct,
        'correct_answer': correct_answer,
        'user_answer': button_label
    }
    st.session_state.vcv_practice_trials_count += 1
    if st.session_state.vcv_practice_trials_count >= N_PRACTICE_TRIALS:
      # Practice finished.
      st.session_state.vcv_practice_completed = True
      st.session_state.vcv_is_practice_trial = False
      st.session_state.vcv_play_button_disabled = False # Enables "Start test"
      st.rerun()

    # LOOP: Queue another practice trial immediately if not done.
    # Queue next practice audio
    active_set = _get_active_consonant_set()
    random_consonant = random.choice(list(active_set.keys()))
    st.session_state.vcv_pending_audio = {
        'consonant': random_consonant,
        'snr': PRACTICE_SNR_DB,
        'ear': 'both'
    }

    st.rerun()

  if st.session_state.vcv_test_mode == 'Adaptive':
    is_correct = button_label == correct_answer
    ear, target_consonant = st.session_state.vcv_last_condition_key
    st.session_state.vcv_responses.append((
        ear, st.session_state.last_played_audio, target_consonant, button_label,
        is_correct, response_time_s, st.session_state.vcv_last_snr
    ))

    # Rename the saved WAV to include the actual response.
    _rename_saved_wav(button_label)

    estimator = st.session_state.vcv_estimators[
        st.session_state.vcv_last_condition_key
    ]
    estimator.update(st.session_state.vcv_last_snr, is_correct)

    if (len(st.session_state.vcv_responses) >=
        st.session_state.vcv_n_total_trials):
      complete_test()
    else:
      prepare_next_trial()
  else:
    st.session_state.vcv_responses.append((
        st.session_state.vcv_current_ear,
        st.session_state.last_played_audio,
        button_label,
        correct_answer,
        response_time_s,
        st.session_state.vcv_last_snr
    ))

    # Rename the saved WAV to include the actual response.
    _rename_saved_wav(button_label)

    test_finished = False
    if st.session_state.vcv_merge_lr:
      st.session_state.vcv_play_count += 1
      if (st.session_state.vcv_play_count >=
          st.session_state.vcv_n_total_trials // 2):
        test_finished = True
    else:
      st.session_state.vcv_completed_stimuli_current_ear += 1
      if (st.session_state.vcv_completed_stimuli_current_ear >=
          st.session_state.vcv_n_total_trials // 2):
        if st.session_state.vcv_current_ear == 'left':
          st.session_state.vcv_current_ear = 'right'
          st.session_state.vcv_completed_stimuli_current_ear = 0
        else:
          test_finished = True

    if test_finished:
      complete_test()
    else:
      play_next_constant()

def create_progress_bar():
  """Creates a progress bar for the consonant confusion test."""
  if st.session_state.vcv_test_mode == 'Adaptive':
    total_tests = st.session_state.vcv_n_total_trials
    current_progress = len(st.session_state.vcv_responses)
  else:
    n_per_ear = st.session_state.vcv_n_total_trials // 2
    if st.session_state.vcv_merge_lr:
      total_tests = n_per_ear
      current_progress = st.session_state.vcv_play_count
    else:
      total_tests = n_per_ear * 2
      ear_offset = (
          n_per_ear if st.session_state.vcv_current_ear == 'right' else 0
      )
      current_progress = (
          ear_offset + st.session_state.vcv_completed_stimuli_current_ear
      )

  # Override progress bar for practice session.
  if not st.session_state.vcv_practice_completed:
    total_tests = N_PRACTICE_TRIALS
    current_progress = st.session_state.vcv_practice_trials_count

  if total_tests > 0:
    progress_percent = min(1.0, current_progress / total_tests)
  else:
    progress_percent = 0.0

  st.progress(progress_percent)
  if not st.session_state.vcv_practice_completed:
    st.write(f'Practice Progress: {current_progress}/{total_tests}')
  else:
    st.write(
      f'{int(progress_percent * 100)}% Complete '
      f'({current_progress}/{total_tests})'
    )

def create_test_instructions():
  if st.session_state.vcv_test_mode == 'Adaptive':
    st.write("Click the 'Practice' button to begin.")
    st.write('1. Click **Next** to hear a sound.')
    st.write(
        '2. Click the consonant button that matches the sound you heard. '
        '**If you are unsure, please make your best guess.**'
    )
    st.info(
        'Note: The test is designed so that you will get approximately '
        '50% of the answers correct. It is normal to find it difficult!'
    )
  else:
    st.write("Click the 'Start test' button to begin. Then click the "
             'consonant button that best matches the sound you hear.')

def _on_volume_change():
  """Copies select slider label to persistent dB key."""
  label = st.session_state.vcv_vol_widget
  st.session_state.vcv_volume_db_spl = (
    HEARING_PROFILES[label]
  )

def display_feedback():
  """Displays feedback from the last practice trial."""
  fb = st.session_state.get('vcv_practice_feedback')
  if not fb:
    return

  if fb['is_correct']:
    color = 'green'
    msg = 'Correct!'
    icon = '✅'
  else:
    color = 'red'
    msg = f"Incorrect. The answer was {fb['correct_answer']}."
    icon = '❌'

  st.markdown(
      f"""
      <div style="text-align: center; color: {color}; font-size: 1.2em;
                  font-weight: bold; margin-top: 10px;">
          {icon} {msg}
      </div>
      """,
      unsafe_allow_html=True
  )


def create_response_button_grid():
  """Renders volume control and button circle."""
  volume_disabled = (
    st.session_state.vcv_freeze_settings
    and not st.session_state.vcv_is_practice_trial
  )

  current_label = _DB_TO_HEARING_PROFILE.get(
    st.session_state.vcv_volume_db_spl, 'Normal hearing'
  )

  # Hearing profile radio buttons.
  st.radio(
    'User hearing profile',
    options=list(HEARING_PROFILES.keys()),
    index=list(HEARING_PROFILES.keys()).index(current_label),
    key='vcv_vol_widget',
    on_change=_on_volume_change,
    disabled=volume_disabled,
    horizontal=True,
    help=(
      'Adjusts the volume of the speech.'
    )
  )

  create_unified_response_grid()
  display_feedback()


def _inject_dynamic_circle_css(labels: list[str]):
  """Injects inline CSS to position N buttons evenly around a circle."""
  # Match the static CSS container: 500×480 px, button 70×70 px.
  # Centre derived from Practice button (120×120 at left:190, top:175).
  cx, cy = 250, 235  # Centre of the ring (button-centre coords).
  radius = 175       # Distance from centre to button centre.
  btn_half = 35      # Half the button width/height (70/2).

  rules = []
  n = len(labels)
  for i, label in enumerate(labels):
    angle_rad = 2 * math.pi * i / n - math.pi / 2  # Start from top.
    x = cx + radius * math.cos(angle_rad) - btn_half
    y = cy + radius * math.sin(angle_rad) - btn_half
    rules.append(
        f'.st-key-vcv_circle .st-key-btn_{label} {{\n'
        f'  position: absolute !important;\n'
        f'  left: {x:.0f}px;\n'
        f'  top: {y:.0f}px;\n'
        f'}}'
    )

  css = '<style>\n' + '\n'.join(rules) + '\n</style>'
  st.markdown(css, unsafe_allow_html=True)


def create_unified_response_grid():
  """Creates consonant buttons in a circular layout."""
  # Buttons disabled if test completed or not started.
  btns_disabled = (
    st.session_state.vcv_test_completed
    or not st.session_state.vcv_freeze_settings
    or not st.session_state.vcv_play_button_disabled
  )

  with st.container(key='vcv_circle'):
    # Center button: Practice or Start test.
    if not st.session_state.vcv_practice_completed:
      # --- PRACTICE BUTTON STATE ---
      if not st.session_state.vcv_play_button_disabled:
        practice_btn_key = (
          'vcv_btn_practice_enabled'
        )
      else:
        practice_btn_key = (
          'vcv_btn_practice_disabled'
        )

      if st.button(
        'Practice',
        key=practice_btn_key,
        disabled=(
          st.session_state.vcv_play_button_disabled
        ),
        icon=':material/play_arrow:'
      ):
        st.session_state.vcv_play_button_disabled = (
          True
        )
        st.session_state.vcv_freeze_settings = True
        st.session_state.vcv_is_practice_trial = True
        st.session_state.vcv_practice_trials_count = 0
        st.session_state.vcv_practice_feedback = None

        active_set = _get_active_consonant_set()
        random_consonant = random.choice(
          list(active_set.keys())
        )
        st.session_state.vcv_pending_audio = {
          'consonant': random_consonant,
          'snr': PRACTICE_SNR_DB,
          'ear': 'both'
        }
        st.rerun()

    else:
      # --- START TEST BUTTON STATE ---
      if not st.session_state.vcv_play_button_disabled:
        play_button_key = 'vcv_btn_start_enabled'
      else:
        play_button_key = 'vcv_btn_start_disabled'
      if st.button(
        'Start test',
        key=play_button_key,
        disabled=(
          st.session_state.vcv_play_button_disabled
        )
      ):
        # Stop practice mode, start real test.
        st.session_state.vcv_is_practice_trial = (
          False
        )
        st.session_state.vcv_play_button_disabled = (
          True
        )
        st.session_state.vcv_freeze_settings = True
        st.session_state.vcv_practice_feedback = None

        # Create WAV save directory for NAL + LOCAL.
        if _is_nal_local():
          ts = datetime.now().strftime('%Y%m%d_%H%M%S')
          wav_dir = os.path.join(
              'local_results', f'vcv_wavs_{ts}'
          )
          os.makedirs(wav_dir, exist_ok=True)
          st.session_state.vcv_wav_save_dir = wav_dir
          print(f'WAV save directory: {wav_dir}')

        if (st.session_state.vcv_test_mode
            == 'Adaptive'):
          prepare_next_trial()
        else:
          play_next_constant()
        st.rerun()

    # Inject dynamic circle positions when using a custom subset.
    circle_order = _get_active_circle_order()
    if st.session_state.vcv_stimuli_type == 'Custom' and circle_order:
      _inject_dynamic_circle_css(circle_order)

    # Consonant buttons arranged in a circle.
    for label in circle_order:
      if st.button(
        label,
        key=f'btn_{label}',
        disabled=btns_disabled
      ):
        handle_response_button_click(label)




def _process_and_play_vcv(clean_file_path: str, snr_db: float, ear: str):
  """
  Processes and plays a VCV stimulus, applying scaling and adding noise.
  """
  # Debug output for sanity-checking during local testing.
  consonant = get_correct_answer(clean_file_path)
  trial_num = len(st.session_state.vcv_responses) + 1
  is_practice = st.session_state.vcv_is_practice_trial
  floor = CONSONANT_SNR_FLOOR_DB.get(consonant, '?')
  mode = 'PRACTICE' if is_practice else 'TEST'
  print(
      f'[VCV {mode}] Trial {trial_num} | '
      f'Consonant: {consonant} | '
      f'SNR: {snr_db:+.1f} dB | '
      f'Floor: {floor} dB | '
      f'Ear: {ear}'
  )

  # 1. Determine the correct reference SPL for scaling.
  if st.session_state.vcv_stimuli_type == 'Synthetic':
    ref_spl = (calibration.REFERENCE_VCV_DB_SPL +
               calibration.SYNTHETIC_VCV_DB_SPL_OFFSET)
  else:
    # Human and Custom both use the standard reference.
    ref_spl = calibration.REFERENCE_VCV_DB_SPL

  # 2. Get the scaled audio data.
  try:
    target_spl = st.session_state.vcv_volume_db_spl
    scaled_data, sample_rate = common.get_scaled_vcv_data(
        clean_file_path,
        target_db_spl=target_spl,
        ref_db_spl=ref_spl
    )
    # If the stimulus is synthetic, add the leading silence.
    if st.session_state.vcv_stimuli_type == 'Synthetic':
      scaled_data = common.prepend_silence(
          scaled_data,
          sample_rate,
          calibration.SYNTHETIC_VCV_LEADING_SILENCE_S
      )
  except (FileNotFoundError, TypeError, ValueError):
    # Errors are already logged by the function, so just stop.
    st.session_state.response_buttons_disabled = True
    st.session_state.vcv_play_button_disabled = True
    return

  # 3. Mix the scaled audio with noise.
  # We calculate the target RMS amplitude corresponding to the target SPL
  # (set by user) to ensures the TOTAL output level (Speech + Noise) is always
  # the target dB SPL.

  mixed_audio, _ = audio_synthesis.mix_vcv_with_noise(
      snr_db=snr_db,
      ear=ear,
      noise_type=NOISE_TYPE_FOR_SYNTHESIS,
      audio_data=scaled_data,
      sample_rate=sample_rate,
      ramp_duration_s=NOISE_RAMP_DURATION_S
  )

  # 4. Play the final, mixed audio.
  silence = np.zeros((int(sample_rate * 0.1), 2), dtype=mixed_audio.dtype)
  final_audio = np.concatenate([silence, mixed_audio], axis=0)

  with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpf:
    wavfile.write(tmpf.name, sample_rate, final_audio)
    common.autoplay_audio(tmpf.name)
    st.session_state.last_played_audio = clean_file_path
    st.session_state.vcv_last_snr = snr_db
    st.session_state.vcv_tone_start_time = time.time()

    # Save a copy of the played audio for NAL study mode.
    wav_dir = st.session_state.vcv_wav_save_dir
    if wav_dir and not is_practice:
      snr_str = f'{snr_db:+.1f}'.replace('+', 'p').replace('-', 'n')
      save_name = (
          f'trial_{trial_num:03d}_target_{consonant}'
          f'_response_PENDING_snr_{snr_str}dB_{ear}.wav'
      )
      save_path = os.path.join(wav_dir, save_name)
      shutil.copy2(tmpf.name, save_path)
      st.session_state.vcv_last_saved_wav_path = save_path
      print(f'  Saved trial WAV: {save_name}')


def _is_nal_mode():
  """Returns True if the app is running in NAL study mode."""
  return st.session_state.get('app_target_audience') == 'NAL'


def _is_nal_local():
  """Returns True if NAL mode AND running locally."""
  return (_is_nal_mode() and
          st.session_state.get('is_running_locally', False))


def _display_custom_stimuli_uploader(disabled: bool):
  """Shows the ZIP upload interface for custom VCV stimuli."""
  if disabled:
    st.info('Custom stimuli settings are locked during the test.')
    return

  uploaded_file = st.file_uploader(
      'Upload VCV stimuli (ZIP file)',
      type=['zip'],
      key='vcv_custom_zip_uploader',
      help=(
          'Upload a ZIP file containing subfolders named by VCV '
          'token (e.g., ABA/, ADA/). Each subfolder should contain WAVs.'
      ),
  )

  if (uploaded_file is not None and
      not st.session_state.vcv_custom_stimuli_loaded):
    with st.spinner('Extracting and validating stimuli...'):
      # Clean up any previous temp dir.
      custom_vcv_loader.cleanup_temp_dir(
          st.session_state.vcv_custom_wav_dir
      )

      stimuli, messages = custom_vcv_loader.extract_and_validate_zip(
          uploaded_file.getvalue()
      )

      if stimuli:
        st.session_state.vcv_custom_wav_dir = None
        # The extract function created a temp dir internally;
        # retrieve it from the first file path.
        first_path = next(iter(stimuli.values()))[0]
        # Walk up to find the temp root (parent of the VCV folder).
        vcv_folder = os.path.dirname(first_path)
        temp_root = os.path.dirname(vcv_folder)
        # Check if there's a wrapper folder.
        if os.path.basename(temp_root).startswith('vcv_custom_'):
          st.session_state.vcv_custom_wav_dir = temp_root
        else:
          st.session_state.vcv_custom_wav_dir = temp_root

        st.session_state.vcv_custom_consonants = sorted(stimuli.keys())
        st.session_state.vcv_custom_stimuli_loaded = True
        st.session_state.vcv_custom_load_errors = messages

        # Rebuild the base files now that we have custom stimuli.
        st.session_state.base_files_by_consonant = _get_all_base_names()
        # Re-initialise estimators for the new consonant set.
        st.session_state.vcv_estimators = None
        initialize_estimators()
      else:
        st.session_state.vcv_custom_load_errors = messages

  # Display status messages.
  if st.session_state.vcv_custom_load_errors:
    for msg in st.session_state.vcv_custom_load_errors:
      if msg.startswith('✅'):
        st.success(msg)
      elif msg.startswith('❌'):
        st.error(msg)
      else:
        st.warning(msg)


def display_settings():
  """Displays the settings for the consonant confusion test."""
  st.subheader(common.SETTINGS_TITLE)
  st.write(common.SETTINGS_STRING)
  is_nal = _is_nal_mode()
  settings_disabled = st.session_state.vcv_freeze_settings or is_nal

  st.radio(
      'Test Method:',
      ['Adaptive', 'Constant SNR'],
      key='vcv_test_mode',
      horizontal=True,
      disabled=settings_disabled,
      on_change=reset_results_only,
  )
  st.radio(
      'Select Stimuli Type:',
      ['Human', 'Synthetic', 'Custom'],
      key='vcv_stimuli_type',
      horizontal=True,
      disabled=settings_disabled,
      on_change=reset_results_only,
  )
  # Custom stimuli: ZIP upload interface.
  if st.session_state.vcv_stimuli_type == 'Custom':
    _display_custom_stimuli_uploader(settings_disabled)
  st.toggle(
      'Merge L/R',
      key='vcv_merge_lr',
      help=common.MERGE_LR_HELP,
      disabled=settings_disabled,
      on_change=reset_results_only,
  )

  st.slider('Total number of trials:', 10, 200,
            key='vcv_n_total_trials',
            step=10,
            disabled=settings_disabled,
            on_change=reset_results_only)
  if st.session_state.vcv_test_mode == 'Constant SNR':
    st.write('Select SNR levels to test:')
    def on_snr_change():
      st.session_state.vcv_snr_levels = [
          snr for snr in SNR_OPTIONS_DB if st.session_state[f'snr_{snr}']
      ]

    cols = st.columns(len(SNR_OPTIONS_DB))
    for i, snr in enumerate(SNR_OPTIONS_DB):
      with cols[i]:
        st.checkbox(
            snr, key=f'snr_{snr}',
            value=snr in st.session_state.vcv_snr_levels,
            on_change=on_snr_change, disabled=settings_disabled
        )

def play_next_constant():
  """Progression logic for Constant SNR mode."""
  snr = float(random.choice(st.session_state.vcv_snr_levels))
  clean_path = get_random_audio_file_for_practice()
  if clean_path:
    _process_and_play_vcv(clean_path, snr, st.session_state.vcv_current_ear)

def schedule_next_trial(estimators: dict) -> tuple[tuple[str, str], float]:
  """
  Selects the next trial using Weighted Random Sampling based on uncertainty.
  """
  candidates = []
  weights = []

  for key, estimator in estimators.items():
    _, uncertainty = estimator.get_estimate()

    # We raise uncertainty to a power (e.g., 2) to exaggerate the differences.
    # This makes high uncertainty items MUCH more likely to be picked,
    # but still allows others a chance.
    weight = uncertainty ** 2
    candidates.append(key)
    weights.append(weight)

  if candidates:
    selected_key = random.choices(candidates, weights=weights, k=1)[0]
  else:
    return random.choice(list(estimators.keys())), 0.0

  selected_estimator = estimators[selected_key]
  next_snr = selected_estimator.get_next_snr()

  # Use the per-consonant SNR floor based on class.
  consonant = selected_key[1]
  min_snr = CONSONANT_SNR_FLOOR_DB[consonant]

  return selected_key, np.clip(
      next_snr,
      min_snr,
      bayesian_vcv_estimator.MAX_SNR_DB,
  )



def create_main_demo():
  set_initial_demo_state()

  create_intro_text()
  display_settings()
  common.display_preparation()
  create_test_instructions()

  if not st.session_state.vcv_test_completed:
    create_response_button_grid()

  # Progress bar renders after buttons so that the
  # current rerun's handler updates are reflected.
  create_progress_bar()

  if st.session_state.vcv_pending_audio:
    details = st.session_state.vcv_pending_audio
    clean_path = get_stimulus_for_consonant(details['consonant'])
    if clean_path:
      _process_and_play_vcv(clean_path, details['snr'], details['ear'])
    else:
      st.error('Audio file missing.')
    st.session_state.vcv_pending_audio = None

  # Display the results if the test is completed.
  if st.session_state.vcv_test_completed:
    active_set = _get_active_consonant_set()
    active_ordered = [
        c for c in ORDERED_LABELS if c in active_set
    ]
    if st.session_state.vcv_test_mode == 'Adaptive':
      vcv_results.display_adaptive_results(
          st.session_state.vcv_final_estimates,
          st.session_state.vcv_df,
          st.session_state.vcv_n_total_trials,
          st.session_state.vcv_confusion_results,
          active_ordered
      )
      if st.session_state.app_target_audience != 'NAL':
        vcv_results.display_adaptive_interpretation()
    else:
      vcv_results.display_results(
          st.session_state.vcv_results_left,
          st.session_state.vcv_results_right,
          list(active_set.keys()),
          st.session_state.vcv_df,
          st.session_state.vcv_merge_lr,
          st.session_state.vcv_n_total_trials // 2
      )
      if st.session_state.app_target_audience != 'NAL':
        vcv_results.display_constant_interpretation()
