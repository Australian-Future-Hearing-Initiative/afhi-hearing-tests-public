"""Safety-net tests for demo_vcv.py functions at risk of regression.

These tests cover existing behaviour of:
  - get_correct_answer()
  - _get_all_base_names()
  - schedule_next_trial()

They are written *before* the custom-stimuli feature to catch regressions.
"""

from unittest.mock import patch, MagicMock

import pytest

import bayesian_vcv_estimator
import demo_vcv


# ---------------------------------------------------------------------------
# get_correct_answer
# ---------------------------------------------------------------------------

class TestGetCorrectAnswer:
  """Tests for demo_vcv.get_correct_answer()."""

  @pytest.mark.parametrize('filename, expected', [
      ('VCV_aba_1_60SNR.wav', 'B'),
      ('VCV_ada_2_60SNR.wav', 'D'),
      ('VCV_aga_1_60SNR.wav', 'G'),
      ('VCV_aka_1_60SNR.wav', 'K'),
      ('VCV_ana_1_60SNR.wav', 'N'),
      ('VCV_asa_1_60SNR.wav', 'S'),
      ('VCV_asha_1_60SNR.wav', 'SH'),
      ('VCV_ata_1_60SNR.wav', 'T'),
      ('VCV_ava_1_60SNR.wav', 'V'),
      ('VCV_aza_1_60SNR.wav', 'Z'),
  ])
  def test_all_ten_consonants_by_basename(self, filename, expected):
    """Each of the 10 standard VCV filenames resolves to its consonant."""
    assert demo_vcv.get_correct_answer(filename) == expected

  @pytest.mark.parametrize('filename, expected', [
      ('aba', 'B'),
      ('ada', 'D'),
      ('aga', 'G'),
      ('aka', 'K'),
      ('ana', 'N'),
      ('asa', 'S'),
      ('asha', 'SH'),
      ('ata', 'T'),
      ('ava', 'V'),
      ('aza', 'Z'),
  ])
  def test_bare_vcv_names(self, filename, expected):
    """Bare VCV token names (no prefix/suffix) also resolve."""
    assert demo_vcv.get_correct_answer(filename) == expected

  def test_full_path(self):
    """A full file path should still resolve correctly (uses basename)."""
    path = '/some/deep/path/to/stimuli/VCV_aba_1_60SNR.wav'
    assert demo_vcv.get_correct_answer(path) == 'B'

  def test_unknown_name_returns_none(self):
    """An unrecognised filename should return None."""
    assert demo_vcv.get_correct_answer('unknown_file.wav') is None

  def test_empty_string_returns_none(self):
    """An empty string should return None."""
    assert demo_vcv.get_correct_answer('') is None

  def test_synthetic_filename(self):
    """Synthetic stimuli filenames (bare VCV name) also resolve."""
    assert demo_vcv.get_correct_answer('asha.wav') == 'SH'


# ---------------------------------------------------------------------------
# _get_all_base_names
# ---------------------------------------------------------------------------

class TestGetAllBaseNames:
  """Tests for demo_vcv._get_all_base_names() (mocked filesystem)."""

  @patch('demo_vcv.st')
  @patch('demo_vcv.glob.glob')
  @patch('demo_vcv.os.path.isdir', return_value=True)
  def test_human_stimuli_bucketing(
      self, _, mock_glob, mock_st
  ):
    """Human stimuli files are bucketed by consonant correctly."""
    mock_st.session_state.vcv_stimuli_type = 'Human'
    mock_glob.return_value = [
        '/path/stimuli/clean_standardised/VCV_aba_1_60SNR.wav',
        '/path/stimuli/clean_standardised/VCV_aba_2_60SNR.wav',
        '/path/stimuli/clean_standardised/VCV_ada_1_60SNR.wav',
        '/path/stimuli/clean_standardised/VCV_asha_1_60SNR.wav',
    ]

    result = demo_vcv._get_all_base_names()  # pylint: disable=protected-access

    # Check that files are bucketed into the right consonants.
    assert len(result['B']) == 2
    assert 'VCV_aba_1_60SNR' in result['B']
    assert 'VCV_aba_2_60SNR' in result['B']
    assert len(result['D']) == 1
    assert len(result['SH']) == 1
    # All other consonants should have empty lists.
    for c in ['G', 'K', 'N', 'S', 'T', 'V', 'Z']:
      assert result[c] == []

  @patch('demo_vcv.st')
  @patch('demo_vcv.glob.glob')
  @patch('demo_vcv.os.path.isdir', return_value=True)
  def test_synthetic_stimuli_bucketing(
      self, _, mock_glob, mock_st
  ):
    """Synthetic stimuli files (bare VCV names) are bucketed correctly."""
    mock_st.session_state.vcv_stimuli_type = 'Synthetic'
    mock_glob.return_value = [
        '/path/stimuli/synthetic/aba.wav',
        '/path/stimuli/synthetic/ada.wav',
    ]

    result = demo_vcv._get_all_base_names()  # pylint: disable=protected-access

    assert len(result['B']) == 1
    assert 'aba' in result['B']
    assert len(result['D']) == 1
    assert 'ada' in result['D']

  @patch('demo_vcv.st')
  @patch('demo_vcv.os.path.isdir', return_value=False)
  def test_missing_directory_returns_empty(self, _, mock_st):
    """Returns empty dict when the stimuli directory does not exist."""
    mock_st.session_state.vcv_stimuli_type = 'Human'

    result = demo_vcv._get_all_base_names()  # pylint: disable=protected-access

    assert result == {}
    mock_st.error.assert_called_once()

  @patch('demo_vcv.st')
  @patch('demo_vcv.glob.glob')
  @patch('demo_vcv.os.path.isdir', return_value=True)
  def test_all_consonants_present_in_output(
      self, _, mock_glob, mock_st
  ):
    """Output dict always has keys for all 10 consonants."""
    mock_st.session_state.vcv_stimuli_type = 'Human'
    mock_glob.return_value = []

    result = demo_vcv._get_all_base_names()  # pylint: disable=protected-access

    expected_keys = {'B', 'D', 'G', 'K', 'N', 'S', 'SH', 'T', 'V', 'Z'}
    assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# schedule_next_trial
# ---------------------------------------------------------------------------

class TestScheduleNextTrial:
  """Tests for demo_vcv.schedule_next_trial()."""

  @pytest.fixture
  def mock_estimators(self):
    """Creates a minimal dict of mock estimators for all 10 consonants."""
    estimators = {}
    for consonant in bayesian_vcv_estimator.CONSONANT_LABELS:
      key = ('both', consonant)
      estimator = MagicMock()
      # Default: moderate uncertainty, 0 dB estimate, 6 samples.
      estimator.get_estimate.return_value = (0.0, 10.0)
      estimator.get_next_snr.return_value = 0.0
      estimator.history = [0.0] * 6
      estimators[key] = estimator
    return estimators

  def test_returns_valid_key(self, mock_estimators):
    """Returned key should be one of the estimator keys."""
    key, _ = demo_vcv.schedule_next_trial(mock_estimators)
    assert key in mock_estimators

  def test_returned_key_is_ear_consonant_tuple(self, mock_estimators):
    """Returned key should be a (ear, consonant) tuple."""
    key, _ = demo_vcv.schedule_next_trial(mock_estimators)
    ear, consonant = key
    assert ear == 'both'
    assert consonant in bayesian_vcv_estimator.CONSONANT_LABELS

  def test_snr_within_bounds(self, mock_estimators):
    """Returned SNR should be within [floor, MAX_SNR_DB]."""
    # Set a specific estimator to return a very low SNR.
    mock_estimators[('both', 'B')].get_estimate.return_value = (0.0, 100.0)
    mock_estimators[('both', 'B')].get_next_snr.return_value = -100.0

    # Run multiple times since selection is random.
    for _ in range(50):
      key, snr = demo_vcv.schedule_next_trial(mock_estimators)
      consonant = key[1]
      floor = bayesian_vcv_estimator.CONSONANT_SNR_FLOOR_DB[consonant]
      assert snr >= floor, (
          f'SNR {snr} below floor {floor} for {consonant}'
      )
      assert snr <= bayesian_vcv_estimator.MAX_SNR_DB, (
          f'SNR {snr} above max {bayesian_vcv_estimator.MAX_SNR_DB}'
      )

  def test_high_uncertainty_preferred(self, mock_estimators):
    """Consonant with much higher uncertainty should be selected more often."""
    # Give one consonant very high uncertainty.
    mock_estimators[('both', 'B')].get_estimate.return_value = (0.0, 100.0)
    mock_estimators[('both', 'B')].get_next_snr.return_value = 0.0
    # Give all others very low uncertainty.
    for key, est in mock_estimators.items():
      if key != ('both', 'B'):
        est.get_estimate.return_value = (0.0, 0.1)

    selections = []
    for _ in range(200):
      key, _ = demo_vcv.schedule_next_trial(mock_estimators)
      selections.append(key)

    b_count = sum(1 for k in selections if k == ('both', 'B'))
    # With uncertainty 100 vs 0.1, B should be selected overwhelmingly.
    assert b_count > 150, (
        f'Expected B to be selected >150/200 times, got {b_count}'
    )

  def test_snr_clipped_to_floor(self, mock_estimators):
    """If estimator suggests SNR below the consonant's floor, it's clipped."""
    # B has floor_db = -8.0. Set its next_snr well below that.
    mock_estimators[('both', 'B')].get_estimate.return_value = (0.0, 100.0)
    mock_estimators[('both', 'B')].get_next_snr.return_value = -50.0
    # Make B the only likely selection.
    for key, est in mock_estimators.items():
      if key != ('both', 'B'):
        est.get_estimate.return_value = (0.0, 0.001)

    key, snr = demo_vcv.schedule_next_trial(mock_estimators)
    assert key == ('both', 'B')
    assert snr == bayesian_vcv_estimator.CONSONANT_SNR_FLOOR_DB['B']

  def test_snr_clipped_to_max(self, mock_estimators):
    """If estimator suggests SNR above MAX_SNR_DB, it's clipped."""
    mock_estimators[('both', 'B')].get_estimate.return_value = (0.0, 100.0)
    mock_estimators[('both', 'B')].get_next_snr.return_value = 999.0
    for key, est in mock_estimators.items():
      if key != ('both', 'B'):
        est.get_estimate.return_value = (0.0, 0.001)

    key, snr = demo_vcv.schedule_next_trial(mock_estimators)
    assert key == ('both', 'B')
    assert snr == bayesian_vcv_estimator.MAX_SNR_DB

  def test_consonants_with_sd_at_or_below_threshold_excluded(
      self, mock_estimators
  ):
    """Consonants at or below the SD threshold are excluded."""
    # Set 'B' (SD=5.0) and 'D' (SD=4.0) as unconverged (> 3.0).
    mock_estimators[('both', 'B')].get_estimate.return_value = (0.0, 5.0)
    mock_estimators[('both', 'D')].get_estimate.return_value = (0.0, 4.0)

    # Set all other consonants to have converged SD <= 3.0.
    for key, est in mock_estimators.items():
      if key not in (('both', 'B'), ('both', 'D')):
        est.get_estimate.return_value = (0.0, 2.8)

    selected_consonants = set()
    for _ in range(100):
      key, _ = demo_vcv.schedule_next_trial(mock_estimators)
      selected_consonants.add(key[1])

    # Only B and D should have been selected; converged ones must be excluded.
    assert selected_consonants.issubset({'B', 'D'})
    assert len(selected_consonants) > 0

  def test_exact_sd_threshold_excluded(self, mock_estimators):
    """Consonant with SD exactly equal to 3.0 is excluded."""
    # Set 'B' to SD=3.0 (exact threshold) and 'D' to SD=3.5 (> 3.0).
    mock_estimators[('both', 'B')].get_estimate.return_value = (0.0, 3.0)
    mock_estimators[('both', 'D')].get_estimate.return_value = (0.0, 3.5)
    for key, est in mock_estimators.items():
      if key not in (('both', 'B'), ('both', 'D')):
        est.get_estimate.return_value = (0.0, 2.0)

    selected = {
        demo_vcv.schedule_next_trial(mock_estimators)[0][1]
        for _ in range(50)
    }
    assert selected == {'D'}

  def test_fallback_when_all_consonants_converged(self, mock_estimators):
    """When all consonants have SD <= 3.0, gracefully sample from all."""
    for est in mock_estimators.values():
      est.get_estimate.return_value = (0.0, 2.0)

    key, snr = demo_vcv.schedule_next_trial(mock_estimators)
    assert key in mock_estimators
    assert isinstance(snr, float)

  def test_low_sample_count_kept_when_sd_converged(
      self, mock_estimators
  ):
    """A low sample count keeps a consonant in the pool."""
    # Set 'B' to SD=2.0 (<= 3.0) but with only 3 samples (< 6).
    mock_estimators[('both', 'B')].get_estimate.return_value = (0.0, 2.0)
    mock_estimators[('both', 'B')].history = [0.0] * 3

    # Set 'D' to SD=4.0 (> 3.0) with 6 samples.
    mock_estimators[('both', 'D')].get_estimate.return_value = (0.0, 4.0)
    mock_estimators[('both', 'D')].history = [0.0] * 6

    # Set all other consonants to converged with 6 samples.
    for key, est in mock_estimators.items():
      if key not in (('both', 'B'), ('both', 'D')):
        est.get_estimate.return_value = (0.0, 2.0)
        est.history = [0.0] * 6

    selected_consonants = set()
    for _ in range(100):
      key, _ = demo_vcv.schedule_next_trial(mock_estimators)
      selected_consonants.add(key[1])

    # B (too few samples) and D (SD still high) stay in the pool.
    assert selected_consonants.issubset({'B', 'D'})
    assert 'B' in selected_consonants

  def test_consonants_excluded_when_both_sd_converged_and_at_least_6_samples(
      self, mock_estimators
  ):
    """Consonant with SD <= 3.0 is excluded once it has at least 6 samples."""
    # Set 'B' to SD=2.0 with 6 samples -> excluded.
    mock_estimators[('both', 'B')].get_estimate.return_value = (0.0, 2.0)
    mock_estimators[('both', 'B')].history = [0.0] * 6

    # Set 'D' to SD=4.0 with 6 samples -> unconverged.
    mock_estimators[('both', 'D')].get_estimate.return_value = (0.0, 4.0)
    mock_estimators[('both', 'D')].history = [0.0] * 6

    # Set all other consonants to converged with 6 samples.
    for key, est in mock_estimators.items():
      if key not in (('both', 'B'), ('both', 'D')):
        est.get_estimate.return_value = (0.0, 2.0)
        est.history = [0.0] * 6

    for _ in range(50):
      key, _ = demo_vcv.schedule_next_trial(mock_estimators)
      assert key[1] == 'D'


@pytest.fixture
def _custom_mode():
  """Sets the stimuli type to Custom for the duration of the test."""
  mock_st = MagicMock()
  mock_st.session_state = {'vcv_stimuli_type': 'Custom'}
  with patch.object(demo_vcv, 'st', mock_st):
    yield


@pytest.fixture
def _human_mode():
  """Sets the stimuli type to Human for the duration of the test."""
  mock_st = MagicMock()
  mock_st.session_state = {'vcv_stimuli_type': 'Human'}
  with patch.object(demo_vcv, 'st', mock_st):
    yield


class TestGetCorrectAnswerCustomStimuli:
  """Tests for get_correct_answer() in Custom stimuli mode."""

  @pytest.mark.usefixtures('_custom_mode')
  def test_resolves_consonant_from_folder(self):
    """Arbitrary WAV filenames resolve via the parent folder name."""
    path = '/tmp/vcv_custom_xyz/ABA/speaker1.wav'
    assert demo_vcv.get_correct_answer(path) == 'B'

  @pytest.mark.usefixtures('_custom_mode')
  def test_wrapper_folder(self):
    """Resolves correctly when a wrapper folder is present."""
    path = '/tmp/vcv_custom_xyz/MyStimuli/ADA/rec_01.wav'
    assert demo_vcv.get_correct_answer(path) == 'D'

  @pytest.mark.usefixtures('_custom_mode')
  def test_case_insensitive_folder(self):
    """Folder names match case-insensitively."""
    for folder_name in ('ASHA', 'asha', 'Asha'):
      path = f'/tmp/vcv_custom_xyz/{folder_name}/speaker1.wav'
      assert demo_vcv.get_correct_answer(path) == 'SH', (
          f'Expected "SH" for folder "{folder_name}".'
      )

  @pytest.mark.usefixtures('_custom_mode')
  def test_ignores_vcv_token_in_filename(self):
    """A VCV token in the filename is ignored; only the folder matters."""
    # File named 'aba_recording.wav' inside the ADA folder → must be D.
    path = '/tmp/vcv_custom_xyz/ADA/aba_recording.wav'
    assert demo_vcv.get_correct_answer(path) == 'D'

  @pytest.mark.usefixtures('_custom_mode')
  @pytest.mark.parametrize('folder,expected', [
      ('ABA', 'B'), ('ADA', 'D'), ('AGA', 'G'), ('AKA', 'K'),
      ('ANA', 'N'), ('ASA', 'S'), ('ASHA', 'SH'), ('ATA', 'T'),
      ('AVA', 'V'), ('AZA', 'Z'),
  ])
  def test_all_standard_consonants_via_folder(self, folder, expected):
    """Every standard consonant resolves from its VCV folder name."""
    path = f'/tmp/vcv_custom_xyz/{folder}/any_name.wav'
    assert demo_vcv.get_correct_answer(path) == expected

  @pytest.mark.usefixtures('_custom_mode')
  def test_practice_feedback_correct_for_custom_stimuli(self):
    """Practice feedback reports the correct consonant for custom paths."""
    custom_path = '/tmp/vcv_custom_xyz/ABA/speaker1.wav'
    correct_answer = demo_vcv.get_correct_answer(custom_path)

    button_label = 'B'
    feedback = {
        'is_correct': button_label == correct_answer,
        'correct_answer': correct_answer,
        'user_answer': button_label,
    }

    assert feedback['correct_answer'] == 'B'
    assert feedback['is_correct'] is True


class TestGetCorrectAnswerStandardStimuli:
  """Tests for get_correct_answer() in Human/Synthetic modes."""

  @pytest.mark.usefixtures('_human_mode')
  def test_standard_human_filename(self):
    """Standard human filename with VCV token resolves correctly."""
    assert demo_vcv.get_correct_answer('VCV_aba_1_60SNR.wav') == 'B'

  @pytest.mark.usefixtures('_human_mode')
  def test_standard_synthetic_filename(self):
    """Synthetic bare-token filename resolves correctly."""
    assert demo_vcv.get_correct_answer('asha.wav') == 'SH'


class _Rerun(Exception):
  """Stands in for Streamlit's rerun, which stops the current run."""


class _SessionState(dict):
  """Dict with attribute access, matching Streamlit session_state."""

  def __getattr__(self, name):
    try:
      return self[name]
    except KeyError as exc:
      raise AttributeError(name) from exc

  def __setattr__(self, name, value):
    self[name] = value


class TestHandleNoSpeechResponse:
  """Tests for NO_SPEECH handling in handle_response_button_click()."""

  def _human_state(self, **overrides):
    """Builds session state for a Human-stimuli VCV trial."""
    state = _SessionState({
        'last_played_audio': 'VCV_aba_1_60SNR.wav',
        'vcv_stimuli_type': 'Human',
        'vcv_tone_start_time': None,
        'vcv_is_practice_trial': False,
        'vcv_practice_trials_count': 0,
        'vcv_practice_feedback': None,
        'vcv_practice_completed': False,
        'vcv_play_button_disabled': True,
        'vcv_custom_consonants': [],
        'vcv_test_mode': 'Adaptive',
        'vcv_last_condition_key': ('left', 'B'),
        'vcv_last_snr': -6.0,
        'vcv_responses': [],
        'vcv_n_total_trials': 90,
        'vcv_merge_lr': False,
        'vcv_current_ear': 'left',
        'vcv_completed_stimuli_current_ear': 0,
        'vcv_estimators': {},
    })
    state.update(overrides)
    return state

  @patch('demo_vcv.st')
  def test_practice_no_speech_is_incorrect(self, mock_st):
    """Practice NO_SPEECH is incorrect and queues another trial."""
    state = self._human_state(vcv_is_practice_trial=True)
    mock_st.session_state = state
    mock_st.rerun.side_effect = _Rerun

    with pytest.raises(_Rerun):
      demo_vcv.handle_response_button_click(
          demo_vcv.NO_SPEECH_RESPONSE
      )

    fb = state.vcv_practice_feedback
    assert fb['is_correct'] is False
    assert fb['user_answer'] == demo_vcv.NO_SPEECH_RESPONSE
    assert fb['correct_answer'] == 'B'
    assert state.vcv_practice_trials_count == 1
    assert state.vcv_pending_audio is not None
    assert state.vcv_pending_audio['snr'] == demo_vcv.PRACTICE_SNR_DB
    mock_st.rerun.assert_called()

  @patch('demo_vcv.prepare_next_trial')
  @patch('demo_vcv._rename_saved_wav')
  @patch('demo_vcv.st')
  def test_adaptive_no_speech_is_a_miss(
      self, mock_st, mock_rename, mock_prepare
  ):
    """Adaptive NO_SPEECH is logged as incorrect and updates ZEST."""
    estimator = MagicMock()
    state = self._human_state(
        vcv_estimators={('left', 'B'): estimator}
    )
    mock_st.session_state = state

    demo_vcv.handle_response_button_click(
        demo_vcv.NO_SPEECH_RESPONSE
    )

    assert len(state.vcv_responses) == 1
    row = state.vcv_responses[0]
    assert row[2] == 'B'
    assert row[3] == demo_vcv.NO_SPEECH_RESPONSE
    assert row[4] is False
    estimator.update.assert_called_once_with(-6.0, False)
    mock_rename.assert_called_once_with(
        demo_vcv.NO_SPEECH_RESPONSE
    )
    mock_prepare.assert_called_once()

  @patch('demo_vcv.play_next_constant')
  @patch('demo_vcv._rename_saved_wav')
  @patch('demo_vcv.st')
  def test_constant_no_speech_advances_trial(
      self, mock_st, mock_rename, mock_play
  ):
    """Constant-SNR NO_SPEECH consumes the trial and continues."""
    state = self._human_state(vcv_test_mode='Constant SNR')
    mock_st.session_state = state

    demo_vcv.handle_response_button_click(
        demo_vcv.NO_SPEECH_RESPONSE
    )

    assert len(state.vcv_responses) == 1
    row = state.vcv_responses[0]
    assert row[2] == demo_vcv.NO_SPEECH_RESPONSE
    assert row[3] == 'B'
    assert state.vcv_completed_stimuli_current_ear == 1
    mock_rename.assert_called_once_with(
        demo_vcv.NO_SPEECH_RESPONSE
    )
    mock_play.assert_called_once()


@patch('demo_vcv.st')
def test_rename_saved_wav_uses_no_speech_token(mock_st, tmp_path):
  """Saved WAV filenames include NO_SPEECH rather than the button text."""
  old_path = tmp_path / (
      'trial_001_target_B_response_PENDING_snr_n6.0dB_left.wav'
  )
  old_path.write_bytes(b'x')
  mock_st.session_state = _SessionState({
      'vcv_last_saved_wav_path': str(old_path),
  })

  demo_vcv._rename_saved_wav(  # pylint: disable=protected-access
      demo_vcv.NO_SPEECH_RESPONSE
  )

  new_path = tmp_path / (
      'trial_001_target_B_response_NO_SPEECH_snr_n6.0dB_left.wav'
  )
  assert new_path.is_file()
  assert not old_path.exists()
  assert mock_st.session_state.vcv_last_saved_wav_path is None


class TestSeparateEarTesting:
  """Tests verifying that left and right ears are tested separately."""

  def test_schedule_next_trial_respects_target_ear_left(self):
    """schedule_next_trial must only pick left ear when target_ear='left'."""
    mock_left = MagicMock()
    mock_left.get_estimate.return_value = (0.0, 5.0)
    mock_left.get_next_snr.return_value = 0.0

    mock_right = MagicMock()
    mock_right.get_estimate.return_value = (0.0, 5.0)
    mock_right.get_next_snr.return_value = 0.0

    estimators = {
        ('left', 'B'): mock_left,
        ('right', 'B'): mock_right,
        ('right', 'D'): mock_right,
    }

    for _ in range(20):
      key, _ = demo_vcv.schedule_next_trial(estimators, target_ear='left')
      assert key[0] == 'left'
      assert key[1] == 'B'

  def test_schedule_next_trial_respects_target_ear_right(self):
    """schedule_next_trial must only pick right ear when target_ear='right'."""
    mock_left = MagicMock()
    mock_left.get_estimate.return_value = (0.0, 5.0)
    mock_left.get_next_snr.return_value = 0.0

    mock_right = MagicMock()
    mock_right.get_estimate.return_value = (0.0, 5.0)
    mock_right.get_next_snr.return_value = 0.0

    estimators = {
        ('left', 'B'): mock_left,
        ('left', 'D'): mock_left,
        ('right', 'P'): mock_right,
    }

    for _ in range(20):
      key, _ = demo_vcv.schedule_next_trial(estimators, target_ear='right')
      assert key[0] == 'right'
      assert key[1] == 'P'

  @patch('demo_vcv.schedule_next_trial')
  @patch('demo_vcv.st')
  def test_prepare_next_trial_uses_current_ear(self, mock_st, mock_schedule):
    """prepare_next_trial passes target_ear based on vcv_current_ear."""
    mock_schedule.return_value = (('left', 'B'), 2.0)
    state = _SessionState({
        'vcv_merge_lr': False,
        'vcv_current_ear': 'left',
        'vcv_estimators': {},
        'vcv_last_condition_key': None,
        'vcv_last_snr': None,
        'vcv_pending_audio': None,
    })
    mock_st.session_state = state

    demo_vcv.prepare_next_trial()

    mock_schedule.assert_called_once_with(
        state.vcv_estimators, target_ear='left'
    )
    assert state.vcv_pending_audio['ear'] == 'left'
    assert state.vcv_pending_audio['consonant'] == 'B'

  @patch('demo_vcv.prepare_next_trial')
  @patch('demo_vcv._rename_saved_wav')
  @patch('demo_vcv.st')
  def test_adaptive_ear_switch_from_left_to_right(
      self, mock_st, unused_rename, mock_prepare
  ):
    """Reaching the left-ear quota moves the test to the right ear."""
    estimator = MagicMock()
    state = _SessionState({
        'last_played_audio': 'VCV_aba_1_60SNR.wav',
        'vcv_stimuli_type': 'Human',
        'vcv_tone_start_time': None,
        'vcv_is_practice_trial': False,
        'vcv_test_mode': 'Adaptive',
        'vcv_last_condition_key': ('left', 'B'),
        'vcv_last_snr': -6.0,
        'vcv_responses': [],
        'vcv_n_total_trials': 10,
        'vcv_merge_lr': False,
        'vcv_current_ear': 'left',
        'vcv_completed_stimuli_current_ear': 4,  # 4 completed out of 5 per ear
        'vcv_ear_switched_notice': False,
        'vcv_estimators': {('left', 'B'): estimator},
    })
    mock_st.session_state = state

    demo_vcv.handle_response_button_click('B')

    assert estimator.update.called
    assert state.vcv_current_ear == 'right'
    assert state.vcv_completed_stimuli_current_ear == 0
    assert state.vcv_ear_switched_notice is True
    mock_prepare.assert_called_once()

  @patch('demo_vcv.complete_test')
  @patch('demo_vcv._rename_saved_wav')
  @patch('demo_vcv.st')
  def test_adaptive_completes_after_both_ears_finish(
      self, mock_st, unused_rename, mock_complete
  ):
    """When right ear completes its quota, complete_test is invoked."""
    estimator = MagicMock()
    state = _SessionState({
        'last_played_audio': 'VCV_aba_1_60SNR.wav',
        'vcv_stimuli_type': 'Human',
        'vcv_tone_start_time': None,
        'vcv_is_practice_trial': False,
        'vcv_test_mode': 'Adaptive',
        'vcv_last_condition_key': ('right', 'B'),
        'vcv_last_snr': -6.0,
        'vcv_responses': [],
        'vcv_n_total_trials': 10,
        'vcv_merge_lr': False,
        'vcv_current_ear': 'right',
        # 4 of 5 trials done on the right ear.
        'vcv_completed_stimuli_current_ear': 4,
        'vcv_ear_switched_notice': False,
        'vcv_estimators': {('right', 'B'): estimator},
    })
    mock_st.session_state = state

    demo_vcv.handle_response_button_click('B')

    assert estimator.update.called
    assert state.vcv_completed_stimuli_current_ear == 5
    mock_complete.assert_called_once()


class TestConvergenceStoppingCondition:
  """Tests that collection stops once every consonant has converged."""

  def test_are_all_consonants_converged_logic(self):
    """Tests convergence across SD and sample-count cases."""
    mock_b = MagicMock()
    mock_b.get_estimate.return_value = (0.0, 2.5)  # converged SD
    mock_b.history = [0.0] * 6
    mock_d = MagicMock()
    # An SD exactly at the threshold counts as converged.
    mock_d.get_estimate.return_value = (0.0, 3.0)
    mock_d.history = [0.0] * 6
    mock_g = MagicMock()
    mock_g.get_estimate.return_value = (0.0, 3.2)  # > 3.0 -> unconverged SD
    mock_g.history = [0.0] * 6

    estimators = {
        ('left', 'B'): mock_b,
        ('left', 'D'): mock_d,
        ('left', 'G'): mock_g,
    }

    # Left ear is not fully converged because G is 3.2
    assert not demo_vcv.are_all_consonants_converged(estimators, 'left')
    assert not demo_vcv.are_all_consonants_converged(estimators, None)

    # Now G SD converges to 2.9, but has only 5 samples (< 6)
    mock_g.get_estimate.return_value = (0.0, 2.9)
    mock_g.history = [0.0] * 5
    assert not demo_vcv.are_all_consonants_converged(estimators, 'left')
    assert not demo_vcv.are_all_consonants_converged(estimators, None)

    # Now G has 6 samples -> all converged
    mock_g.history = [0.0] * 6
    assert demo_vcv.are_all_consonants_converged(estimators, 'left')
    assert demo_vcv.are_all_consonants_converged(estimators, None)

  @patch('demo_vcv.complete_test')
  @patch('demo_vcv._rename_saved_wav')
  @patch('demo_vcv.st')
  def test_adaptive_stops_when_all_consonants_converge_in_merged_mode(
      self, mock_st, unused_rename, mock_complete
  ):
    """Merged mode finishes once every consonant has converged."""
    mock_b = MagicMock()
    mock_b.get_estimate.return_value = (0.0, 2.8)
    mock_b.history = [0.0] * 6
    mock_d = MagicMock()
    mock_d.get_estimate.return_value = (0.0, 2.9)
    mock_d.history = [0.0] * 6

    estimators = {
        ('both', 'B'): mock_b,
        ('both', 'D'): mock_d,
    }

    state = _SessionState({
        'last_played_audio': 'VCV_aba_1_60SNR.wav',
        'vcv_stimuli_type': 'Human',
        'vcv_tone_start_time': None,
        'vcv_is_practice_trial': False,
        'vcv_test_mode': 'Adaptive',
        'vcv_last_condition_key': ('both', 'B'),
        'vcv_last_snr': -6.0,
        'vcv_responses': [],
        'vcv_n_total_trials': 100,  # Far from trial limit (50)
        'vcv_merge_lr': True,
        'vcv_play_count': 10,
        'vcv_stopped_by_convergence': False,
        'vcv_estimators': estimators,
    })
    mock_st.session_state = state

    demo_vcv.handle_response_button_click('B')

    assert mock_b.update.called
    assert state.vcv_stopped_by_convergence is True
    mock_complete.assert_called_once()

  @patch('demo_vcv.prepare_next_trial')
  @patch('demo_vcv._rename_saved_wav')
  @patch('demo_vcv.st')
  def test_adaptive_switches_to_right_ear_when_left_ear_converges_early(
      self, mock_st, unused_rename, mock_prepare
  ):
    """When left ear converges early, test switches to right ear to test it."""
    mock_b_left = MagicMock()
    mock_b_left.get_estimate.return_value = (0.0, 2.5)  # Left converged
    mock_b_left.history = [0.0] * 6
    mock_b_right = MagicMock()
    mock_b_right.get_estimate.return_value = (0.0, 15.0)  # Right unconverged
    mock_b_right.history = [0.0] * 2

    estimators = {
        ('left', 'B'): mock_b_left,
        ('right', 'B'): mock_b_right,
    }

    state = _SessionState({
        'last_played_audio': 'VCV_aba_1_60SNR.wav',
        'vcv_stimuli_type': 'Human',
        'vcv_tone_start_time': None,
        'vcv_is_practice_trial': False,
        'vcv_test_mode': 'Adaptive',
        'vcv_last_condition_key': ('left', 'B'),
        'vcv_last_snr': -6.0,
        'vcv_responses': [],
        'vcv_n_total_trials': 100,  # 50 per ear; currently at only 12
        'vcv_merge_lr': False,
        'vcv_current_ear': 'left',
        'vcv_completed_stimuli_current_ear': 12,
        'vcv_ear_switched_notice': False,
        'vcv_stopped_by_convergence': False,
        'vcv_estimators': estimators,
    })
    mock_st.session_state = state

    demo_vcv.handle_response_button_click('B')

    assert state.vcv_current_ear == 'right'
    assert state.vcv_completed_stimuli_current_ear == 0
    assert state.vcv_ear_switched_notice is True
    mock_prepare.assert_called_once()

  @patch('demo_vcv.complete_test')
  @patch('demo_vcv._rename_saved_wav')
  @patch('demo_vcv.st')
  def test_adaptive_completes_when_both_ears_converged(
      self, mock_st, unused_rename, mock_complete
  ):
    """The test stops once the right ear has converged too."""
    mock_b_left = MagicMock()
    mock_b_left.get_estimate.return_value = (0.0, 2.5)  # Left converged
    mock_b_left.history = [0.0] * 6
    mock_b_right = MagicMock()
    mock_b_right.get_estimate.return_value = (0.0, 2.8)  # Right also converged!
    mock_b_right.history = [0.0] * 6

    estimators = {
        ('left', 'B'): mock_b_left,
        ('right', 'B'): mock_b_right,
    }

    state = _SessionState({
        'last_played_audio': 'VCV_aba_1_60SNR.wav',
        'vcv_stimuli_type': 'Human',
        'vcv_tone_start_time': None,
        'vcv_is_practice_trial': False,
        'vcv_test_mode': 'Adaptive',
        'vcv_last_condition_key': ('right', 'B'),
        'vcv_last_snr': -6.0,
        'vcv_responses': [],
        'vcv_n_total_trials': 100,  # 50 per ear; currently at only 15 on right
        'vcv_merge_lr': False,
        'vcv_current_ear': 'right',
        'vcv_completed_stimuli_current_ear': 15,
        'vcv_ear_switched_notice': False,
        'vcv_stopped_by_convergence': False,
        'vcv_estimators': estimators,
    })
    mock_st.session_state = state

    demo_vcv.handle_response_button_click('B')

    assert state.vcv_stopped_by_convergence is True
    mock_complete.assert_called_once()

  @patch('demo_vcv.complete_test')
  @patch('demo_vcv.st')
  def test_prepare_next_trial_stops_if_all_converged(
      self, mock_st, mock_complete
  ):
    """prepare_next_trial stops when the target ear has converged."""
    mock_b = MagicMock()
    mock_b.get_estimate.return_value = (0.0, 2.5)
    mock_b.history = [0.0] * 6

    estimators = {
        ('both', 'B'): mock_b,
    }

    state = _SessionState({
        'vcv_merge_lr': True,
        'vcv_current_ear': 'both',
        'vcv_stopped_by_convergence': False,
        'vcv_estimators': estimators,
    })
    mock_st.session_state = state

    demo_vcv.prepare_next_trial()

    assert state.vcv_stopped_by_convergence is True
    mock_complete.assert_called_once()



