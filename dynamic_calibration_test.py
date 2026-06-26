"""Integration and unit tests for dynamic calibration and retrieval."""
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from autoeq.frequency_response import FrequencyResponse
from hearing_agent.agent import get_calibration_factors
from hearing_agent.config import AUDIOMETRY_FREQUENCIES
import hearing_agent.retrieval
from hearing_agent.retrieval import github_to_raw_url
from hearing_agent.retrieval import is_bone_conduction_device
from hearing_agent.retrieval import search_autoeq_files


def create_mock_fr(name, smoothed):
  """Helper to create a FrequencyResponse object and populate fields."""
  fr = FrequencyResponse(
      name=name, frequency=AUDIOMETRY_FREQUENCIES, raw=smoothed
  )
  fr.smoothed = np.array(smoothed)
  return fr


class TestDynamicCalibration(unittest.TestCase):
  """Test cases for dynamic calibration pipeline."""

  @patch('hearing_agent.agent.search_autoeq_files')
  @patch('hearing_agent.agent.fetch_frequency_response')
  def test_standard_headphone_calibration(self, mock_fetch, mock_search):
    """Test Case 1: Standard Headphone Calibration"""
    # Set up search mocks
    mock_search.side_effect = (
        lambda name: [
            {
                'name': f'{name} (oratory1990)',
                'database': 'oratory1990',
                'raw_url': f'mock_{name}_url',
            }
        ]
        if name in ['Sony WH-1000XM4', 'Google Pixel Buds Pro 2']
        else []
    )

    # Set up fetch mocks
    def mock_fetch_impl(raw_url, headphone_name):
      del raw_url  # Unused in mock implementation.
      if 'Sony WH-1000XM4' in headphone_name:
        return create_mock_fr(
            headphone_name, [1.5, 0.8, -0.2, -1.0, -2.0, -3.5, -2.5, -1.5]
        )
      elif 'Google Pixel Buds Pro 2' in headphone_name:
        return create_mock_fr(
            headphone_name, [-0.2, 0.2, 0.5, 1.5, 0.8, -1.0, -2.0, -3.0]
        )
      raise ValueError(f'Unknown mock URL/headphone: {headphone_name}')

    mock_fetch.side_effect = mock_fetch_impl

    user_model = 'Sony WH-1000XM4'
    baseline = 'Google Pixel Buds Pro 2'
    res = get_calibration_factors(user_model, baseline)

    self.assertEqual(res['user_headphone'], user_model)
    self.assertEqual(res['baseline_headphone'], baseline)
    self.assertEqual(res['status'], 'success')
    self.assertEqual(res['frequencies'], AUDIOMETRY_FREQUENCIES)
    self.assertEqual(res['bone_conduction_warning'], False)
    self.assertIsNone(res['clipping_warning'])
    self.assertIsNone(res['vetting_warning'])

    # Raw correction = Baseline - User
    # 250Hz: -0.2 - 1.5 = -1.7
    # 500Hz: 0.2 - 0.8 = -0.6
    # 1000Hz: 0.5 - (-0.2) = 0.7
    # 2000Hz: 1.5 - (-1.0) = 2.5
    # 3000Hz: 0.8 - (-2.0) = 2.8
    # 4000Hz: -1.0 - (-3.5) = 2.5
    # 6000Hz: -2.0 - (-2.5) = 0.5
    # 8000Hz: -3.0 - (-1.5) = -1.5
    expected_raw = [-1.7, -0.6, 0.7, 2.5, 2.8, 2.5, 0.5, -1.5]
    self.assertEqual(res['raw_correction_factors_db'], expected_raw)
    self.assertEqual(res['correction_factors_db'], expected_raw)

  @patch('hearing_agent.agent.search_autoeq_files')
  @patch('hearing_agent.agent.fetch_frequency_response')
  def test_bone_conduction_detection(self, mock_fetch, mock_search):
    """Test Case 2: Bone Conduction Device Detection"""
    # Set up search mocks
    mock_search.side_effect = (
        lambda name: [
            {
                'name': f'{name} (rtings)',
                'database': 'rtings',
                'raw_url': f'mock_{name}_url',
            }
        ]
        if name in ['Shokz OpenRun', 'Google Pixel Buds Pro 2']
        else []
    )

    # Set up fetch mocks
    def mock_fetch_impl(raw_url, headphone_name):
      del raw_url  # Unused in mock implementation.
      if 'Shokz OpenRun' in headphone_name:
        return create_mock_fr(headphone_name, [0.0] * 8)
      elif 'Google Pixel Buds Pro 2' in headphone_name:
        return create_mock_fr(headphone_name, [0.0] * 8)
      raise ValueError(f'Unknown mock URL/headphone: {headphone_name}')

    mock_fetch.side_effect = mock_fetch_impl

    bc_model = 'Shokz OpenRun'
    baseline = 'Google Pixel Buds Pro 2'
    res = get_calibration_factors(bc_model, baseline)

    self.assertEqual(res['user_headphone'], bc_model)
    self.assertEqual(res['status'], 'success')
    self.assertEqual(res['bone_conduction_warning'], True)

  @patch('hearing_agent.agent.search_autoeq_files')
  @patch('hearing_agent.agent.fetch_frequency_response')
  def test_safety_calibration_clipping(self, mock_fetch, mock_search):
    """Test Case 3: Safety Calibration Clipping"""
    # Set up search mocks
    mock_search.side_effect = (
        lambda name: [
            {
                'name': f'{name} (oratory1990)',
                'database': 'oratory1990',
                'raw_url': f'mock_{name}_url',
            }
        ]
        if name in ['ultra sensitive phones', 'Google Pixel Buds Pro 2']
        else []
    )

    # Set up fetch mocks
    def mock_fetch_impl(raw_url, headphone_name):
      del raw_url  # Unused in mock implementation.
      if 'ultra sensitive phones' in headphone_name:
        # very high response values to force negative clipping
        # (MIN_CORRECTION = -15)
        return create_mock_fr(
            headphone_name,
            [25.0, 30.0, 28.0, 32.0, 25.0, 20.0, 20.0, 20.0],
        )
      elif 'Google Pixel Buds Pro 2' in headphone_name:
        return create_mock_fr(headphone_name, [0.0] * 8)
      raise ValueError(f'Unknown mock URL/headphone: {headphone_name}')

    mock_fetch.side_effect = mock_fetch_impl

    sensitive_model = 'ultra sensitive phones'
    baseline = 'Google Pixel Buds Pro 2'
    res = get_calibration_factors(sensitive_model, baseline)

    self.assertEqual(res['status'], 'success')
    # Raw correction = 0 - [25, 30, ...] = [-25, -30, ...]
    # Clipped correction should clamp to MIN_CORRECTION_DB = -15.0
    self.assertEqual(res['correction_factors_db'], [-15.0] * 8)
    self.assertIsNotNone(res['clipping_warning'])

  @patch('hearing_agent.agent.search_autoeq_files')
  def test_headphone_not_found(self, mock_search):
    """Test get_calibration_factors returns error if headphone not found."""
    mock_search.return_value = []  # not found

    res = get_calibration_factors(
        'Nonexistent Headphone', 'Google Pixel Buds Pro 2'
    )
    self.assertEqual(res['status'], 'error')
    self.assertIn(
        'Could not find frequency response data', res['error_message']
    )


class TestRetrieval(unittest.TestCase):
  """Test cases for retrieval module."""

  def setUp(self):
    # Reset the cache before each test
    # pylint: disable=protected-access
    hearing_agent.retrieval._AUTOEQ_INDEX_CACHE = None

  def test_is_bone_conduction_device(self):
    self.assertTrue(is_bone_conduction_device('Shokz OpenRun'))
    self.assertTrue(is_bone_conduction_device('AfterShokz Aeropex'))
    self.assertFalse(is_bone_conduction_device('Sony WH-1000XM4'))

  def test_github_to_raw_url(self):
    html_url = (
        'https://github.com/jaakkopasanen/AutoEq/blob/master/results/'
        'oratory1990/over-ear/Sony%20WH-1000XM4/Sony%20WH-1000XM4.csv'
    )
    expected_raw = (
        'https://raw.githubusercontent.com/jaakkopasanen/AutoEq/master/'
        'results/oratory1990/over-ear/Sony%20WH-1000XM4/'
        'Sony%20WH-1000XM4.csv'
    )
    self.assertEqual(github_to_raw_url(html_url), expected_raw)

  @patch('urllib.request.urlopen')
  def test_search_autoeq_files_success(self, mock_urlopen):
    # Mock index markdown file content
    mock_index_content = (
        '- [Sony WH-1000XM4](./oratory1990/over-ear/Sony%20WH-1000XM4) '
        'by oratory1990\n'
        '- [Sony WH-1000XM4]'
        '(./crinacle/GRAS%2043AG-7%20over-ear/Sony%20WH-1000XM4) '
        'by crinacle on GRAS 43AG-7\n'
        '- [Apple AirPods Pro](./oratory1990/in-ear/Apple%20AirPods%20Pro) '
        'by oratory1990\n'
    )

    mock_response = MagicMock()
    mock_response.read.return_value = mock_index_content.encode('utf-8')
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    # Search for Sony WH-1000XM4
    res = search_autoeq_files('Sony WH-1000XM4')
    self.assertEqual(len(res), 1)

    self.assertEqual(res[0]['name'], 'Sony WH-1000XM4')
    self.assertEqual(res[0]['database'], 'oratory1990')
    self.assertEqual(
        res[0]['raw_url'],
        'https://raw.githubusercontent.com/jaakkopasanen/AutoEq/master/'
        'results/oratory1990/over-ear/Sony%20WH-1000XM4/'
        'Sony%20WH-1000XM4.csv',
    )

  @patch('urllib.request.urlopen')
  def test_search_autoeq_files_network_failure(self, mock_urlopen):
    mock_urlopen.side_effect = Exception('Network error')
    res = search_autoeq_files('Sony WH-1000XM4')
    self.assertEqual(res, [])


if __name__ == '__main__':
  unittest.main()
