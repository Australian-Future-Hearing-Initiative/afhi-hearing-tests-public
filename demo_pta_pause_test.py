"""Unit tests for Pause/Resume feature in demo_pta.py."""

import time
import unittest
from unittest.mock import patch

import streamlit as st
import demo_pta


class TestDemoPTAPauseFeature(unittest.TestCase):
  """Verifies Pause/Resume feature by calling demo_pta.py functions."""

  def setUp(self):
    """Reset session state before each test."""
    st.session_state.clear()
    demo_pta.set_initial_demo_state()

  def test_initial_pause_state_variables(self):
    """Verifies that set_initial_demo_state initializes pause variables."""
    demo_pta.set_initial_demo_state()
    self.assertEqual(st.session_state.pta_state, 'Initial')
    self.assertIsNone(st.session_state.pta_pause_start_time)
    self.assertEqual(st.session_state.pta_total_pause_duration_s, 0.0)
    self.assertEqual(st.session_state.pta_pause_count, 0)

  @patch('streamlit.rerun')
  @patch('streamlit.button')
  def test_start_button_pauses_running_test(self, mock_button, mock_rerun):
    """Calling start_button while Running executes pause state transition."""
    st.session_state.pta_method = demo_pta.ADVANCED_METHOD_NAME
    st.session_state.pta_state = 'Running'
    mock_button.return_value = True  # User clicks "Pause the test"

    demo_pta.start_button()

    # Verify button call arguments
    mock_button.assert_called_once_with(
        'Pause the test', key='pta_start_test', icon=':material/pause:',
        disabled=False
    )
    self.assertEqual(st.session_state.pta_state, 'Paused')
    self.assertIsNotNone(st.session_state.pta_pause_start_time)
    self.assertEqual(st.session_state.pta_pause_count, 1)
    mock_rerun.assert_called_once()

  @patch('streamlit.rerun')
  @patch('streamlit.button')
  def test_pausing_leaves_recorded_results_untouched(self, mock_button, _):
    """Transitioning to Paused leaves recorded pta_results history unchanged."""
    st.session_state.pta_method = demo_pta.ADVANCED_METHOD_NAME
    st.session_state.pta_state = 'Running'
    recorded_trials = [
        ('left', 1000.0, 40.0, True, 1.2),
        ('left', 1000.0, 30.0, False, 4.0),
    ]
    st.session_state.pta_results = list(recorded_trials)
    mock_button.return_value = True  # User clicks "Pause the test"

    demo_pta.start_button()

    # Verify recorded trial list is completely untouched
    self.assertEqual(st.session_state.pta_results, recorded_trials)

  @patch('streamlit.rerun')
  @patch('streamlit.button')
  def test_start_button_resumes_paused_test(self, mock_button, mock_rerun):
    """Calling start_button while Paused executes resume state transition."""
    st.session_state.pta_method = demo_pta.ADVANCED_METHOD_NAME
    st.session_state.pta_state = 'Paused'
    st.session_state.pta_pause_start_time = time.time() - 4.0
    st.session_state.pta_total_pause_duration_s = 1.0
    st.session_state.pta_pause_count = 1
    mock_button.return_value = True  # User clicks "Resume the test"

    demo_pta.start_button()

    # Verify button call arguments
    mock_button.assert_called_once_with(
        'Resume the test', key='pta_start_test', icon=':material/play_arrow:',
        disabled=False
    )
    self.assertEqual(st.session_state.pta_state, 'Running')
    self.assertIsNone(st.session_state.pta_pause_start_time)
    # Total paused duration should now be approximately 5.0 seconds (1.0 + 4.0)
    self.assertGreaterEqual(st.session_state.pta_total_pause_duration_s, 4.9)
    mock_rerun.assert_called_once()

  @patch('streamlit.rerun')
  @patch('streamlit.button')
  def test_start_button_hughson_westlake_method(self, mock_button, _):
    """Verifies Hughson-Westlake method renders original Start button logic."""
    st.session_state.pta_method = demo_pta.BASIC_METHOD_NAME
    st.session_state.pta_state = 'Running'
    mock_button.return_value = False

    demo_pta.start_button()

    mock_button.assert_called_once_with(
        'Start the test', key='pta_start_test', icon=':material/play_arrow:',
        disabled=True
    )

  @patch('streamlit.rerun')
  @patch('streamlit.button')
  def test_start_button_enabled_in_completed_state(self, mock_button, _):
    """Calling start_button while Completed displays enabled Start button."""
    st.session_state.pta_method = demo_pta.ADVANCED_METHOD_NAME
    st.session_state.pta_state = 'Completed'
    mock_button.return_value = False

    demo_pta.start_button()

    mock_button.assert_called_once_with(
        'Start the test', key='pta_start_test', icon=':material/play_arrow:',
        disabled=False
    )

  @patch('streamlit.rerun')
  @patch('streamlit.button')
  def test_cancel_button_enabled_and_resets_when_paused(
      self, mock_button, mock_rerun
  ):
    """Calling cancel_button while Paused is enabled and resets to Initial."""
    st.session_state.pta_state = 'Paused'
    st.session_state.pta_pause_start_time = time.time()
    st.session_state.pta_total_pause_duration_s = 10.0
    mock_button.return_value = True  # User clicks "Cancel the test"

    demo_pta.cancel_button()

    # Explicitly verify button rendered with disabled=False while Paused
    mock_button.assert_called_once_with(
        'Cancel the test', key='pta_cancel_test', icon=':material/cancel:',
        disabled=False
    )
    self.assertEqual(st.session_state.pta_state, 'Initial')
    self.assertIsNone(st.session_state.pta_pause_start_time)
    self.assertEqual(st.session_state.pta_total_pause_duration_s, 0.0)
    mock_rerun.assert_called_once()

  @patch('streamlit.rerun')
  @patch('streamlit.button')
  def test_cancel_button_disabled_in_initial_state(self, mock_button, _):
    """Verifies cancel_button is disabled in Initial state."""
    st.session_state.pta_state = 'Initial'
    mock_button.return_value = False

    demo_pta.cancel_button()

    mock_button.assert_called_once_with(
        'Cancel the test', key='pta_cancel_test', icon=':material/cancel:',
        disabled=True
    )

  @patch('streamlit.radio')
  @patch('streamlit.toggle')
  @patch('streamlit.write')
  @patch('streamlit.subheader')
  def test_display_settings_disabled_when_paused(
      self, unused_sub, unused_write, mock_toggle, mock_radio
  ):
    """Calling display_settings while Paused disables settings controls."""
    st.session_state.pta_state = 'Paused'
    st.session_state.toggle_pta_merge_lr = False
    st.session_state.app_target_audience = 'ALL'

    demo_pta.display_settings()

    # Verify toggle was rendered with disabled=True
    mock_toggle.assert_called_once()
    _, toggle_kwargs = mock_toggle.call_args
    self.assertTrue(toggle_kwargs.get('disabled'))

    # Verify radio buttons were rendered with disabled=True
    self.assertGreaterEqual(mock_radio.call_count, 1)
    for call in mock_radio.call_args_list:
      _, radio_kwargs = call
      self.assertTrue(radio_kwargs.get('disabled'))

  @patch('streamlit.info')
  @patch('demo_pta.run_adaptive_pta')
  @patch('demo_pta.create_progress_bar')
  @patch('demo_pta.main_button_layout')
  @patch('demo_pta.demo_button')
  @patch('common.display_preparation')
  @patch('demo_pta.display_settings')
  @patch('demo_pta.create_intro_text')
  def test_create_main_demo_displays_info_when_paused(
      self, unused_intro, unused_settings, unused_prep, unused_demo_btn,
      unused_main_layout, unused_progress, mock_run_adaptive, mock_info
  ):
    """Calling create_main_demo while Paused shows info banner and skips run."""
    st.session_state.pta_state = 'Paused'

    demo_pta.create_main_demo()

    mock_info.assert_called_once_with(
        'Test paused. Click "Resume the test" to continue.'
    )
    mock_run_adaptive.assert_not_called()


if __name__ == '__main__':
  unittest.main()
