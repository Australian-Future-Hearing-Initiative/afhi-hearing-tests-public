"""Extremely minimal unit tests for vcv_results.py, focusing on callability."""

import unittest.mock as mock
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytest

import vcv_results
import common # noqa: E402, F401 pylint: disable=W0611,C0413

@pytest.fixture
def dummy_labels():
  """Provides a list of dummy labels for testing."""
  return ['A', 'B']

@pytest.fixture
def mock_dataframe():
  """Provides a mock DataFrame to avoid issues with real DataFrame creation."""
  df_mock = mock.MagicMock(spec=pd.DataFrame)
  df_mock.copy.return_value = df_mock # Mock copy method.
  # Mimic column check.
  df_mock.columns = ['Response Time (s)']
  df_mock.__contains__.return_value = True # For 'in df.columns' check.
  return df_mock

# --- Test Functions (Highly Mocked) ---

@mock.patch('vcv_results.sns.heatmap') # Mock heatmap directly.
@mock.patch('vcv_results.sns.set')
@mock.patch('vcv_results.plt.subplots') # Mock subplot creation.
def test_create_confusion_matrix_image(
    mock_subplots, mock_sns_set, mock_heatmap,
    dummy_labels # pylint: disable=W0621
):
  """Test create_confusion_matrix_image runs without error."""
  # Arrange: Mock the return value of subplots.
  mock_fig = mock.MagicMock(spec=plt.Figure)
  mock_ax = mock.MagicMock()
  mock_subplots.return_value = (mock_fig, mock_ax)
  dummy_matrix = np.array([[5, 1], [2, 6]])

  # Act & Assert: Call the function and check mocks.
  returned_fig = vcv_results.create_confusion_matrix_image(
      dummy_matrix, dummy_labels
  )
  mock_sns_set.assert_called_once()
  mock_subplots.assert_called_once()
  mock_heatmap.assert_called_once()
  mock_fig.tight_layout.assert_called_once()
  assert returned_fig is mock_fig # Ensure the mocked figure was returned.

# Remove the older test_display_results if it still exists


# --- Test function using st.download_button ---
@patch('vcv_results.create_confusion_matrix_image')
@patch('vcv_results.st.pyplot')
@patch('vcv_results.plt.close')
@patch('vcv_results.common.generate_zip_bytes', return_value=b'mock_zip_bytes')
@patch('vcv_results.st.download_button')
@patch('vcv_results.common.display_email_results_form')
@patch('vcv_results.st') # Patch the whole st module used by vcv_results
def test_display_results(mock_st, # Add mock_st argument
                           mock_display_email, mock_download_button,
                           mock_generate_zip, mock_plt_close, mock_pyplot,
                           mock_create_cm, dummy_labels, mock_dataframe): # pylint: disable=redefined-outer-name
  """Minimal test for display_results focusing on calls, not args."""
  # Dummy results data
  results_left = {'accuracy': 0.8, 'confusion_matrix': [[8, 2], [1, 9]]}
  results_right = {'accuracy': 0.7, 'confusion_matrix': [[7, 3], [2, 8]]}
  mock_fig_left = MagicMock(spec=plt.Figure)
  mock_fig_right = MagicMock(spec=plt.Figure)
  mock_create_cm.side_effect = [mock_fig_left, mock_fig_right]

  # --- Configure mock_st for the first call ---
  mock_col1 = MagicMock()
  mock_col2 = MagicMock()
  mock_st.columns.return_value = [mock_col1, mock_col2]
  mock_st.session_state.is_running_locally = False
  mock_st.session_state.app_target_audience = 'TestAudience'
  mock_st.session_state.vcv_backup_saved = False
  # Ensure the .get call works on the mock session_state
  # Configure .get specifically for the keys we expect
  def mock_session_get(key, default=None):
    if key == 'vcv_backup_saved':
      return getattr(mock_st.session_state, 'vcv_backup_saved', default)
    return default # Default behavior for other keys if needed
  mock_st.session_state.get.side_effect = mock_session_get

  # --- Test Case 1: Separate L/R ---
  vcv_results.display_results(
      results_left, results_right, dummy_labels,
      mock_dataframe, merge_lr=False, n_tests=10
  )
  # Minimal checks - just ensure key functions were called.
  assert mock_create_cm.call_count == 2
  assert mock_pyplot.call_count == 2
  mock_generate_zip.assert_called_once()
  mock_download_button.assert_called_once()
  mock_display_email.assert_called_once()
  assert mock_plt_close.call_count == 2
  # Reset mocks.
  mock_create_cm.reset_mock()
  mock_pyplot.reset_mock()
  mock_generate_zip.reset_mock()
  mock_download_button.reset_mock()
  mock_display_email.reset_mock()
  mock_plt_close.reset_mock()
  mock_st.reset_mock() # Reset the st mock too

  # IMPORTANT: Provide side effect for BOTH internal calls even if merged.
  mock_create_cm.side_effect = [mock_fig_left, mock_fig_right]

  # --- Configure mock_st for the second call ---
  mock_st.columns.return_value = [mock_col1, mock_col2]
  mock_st.session_state.is_running_locally = True # Test different value
  mock_st.session_state.app_target_audience = 'NAL' # Test different value
  mock_st.session_state.vcv_backup_saved = False # Reset flag for this case
  # Re-apply side effect for .get
  mock_st.session_state.get.side_effect = mock_session_get
  # --- Test Case 2: Merged L/R ---
  vcv_results.display_results(
      results_left, results_right, dummy_labels,
      mock_dataframe, merge_lr=True, n_tests=15
  )
  # Minimal checks.
  assert mock_create_cm.call_count == 2
  assert mock_pyplot.call_count == 2
  mock_generate_zip.assert_called_once()
  mock_download_button.assert_called_once()
  mock_display_email.assert_called_once()
  assert mock_plt_close.call_count == 2

@mock.patch('vcv_results.st')
def test_display_interpretation(mock_st):
  """Test display_interpretation runs without error."""
  # Act & Assert.
  vcv_results.display_constant_interpretation()
  # Assert minimal mock calls were made.
  mock_st.write.assert_called()
  mock_st.subheader.assert_called()
  mock_st.image.assert_called_once()
  mock_st.markdown.assert_called_once()
