'''Unit tests for analytics.py.'''

# pylint: disable=protected-access

import numpy as np

import analytics


def test_analyze_results_all_correct():
  '''All responses correct gives accuracy 1.0 and identity matrix.'''
  responses = [('a', 'a'), ('b', 'b'), ('c', 'c')]
  labels = ['a', 'b', 'c']
  result = analytics.analyze_results(responses, labels)
  assert result['correct_answers'] == 3
  assert result['incorrect_answers'] == 0
  assert result['accuracy'] == 1.0
  np.testing.assert_array_equal(
      result['confusion_matrix'], np.eye(3, dtype=int))


def test_analyze_results_all_wrong():
  '''No correct responses gives zero diagonal and accuracy 0.'''
  responses = [('b', 'a'), ('c', 'b'), ('a', 'c')]
  labels = ['a', 'b', 'c']
  result = analytics.analyze_results(responses, labels)
  assert result['correct_answers'] == 0
  assert result['incorrect_answers'] == 3
  assert result['accuracy'] == 0.0
  np.testing.assert_array_equal(
      np.diag(result['confusion_matrix']), np.zeros(3, dtype=int))
  assert result['confusion_matrix'].sum() == 3


def test_analyze_results_mixed():
  '''Mixed inputs populate diagonal and off-diagonal correctly.'''
  # Tuples are (response, correct_answer); 'a' was confused with 'b' once.
  responses = [('a', 'a'), ('b', 'a'), ('b', 'b'), ('c', 'c')]
  labels = ['a', 'b', 'c']
  result = analytics.analyze_results(responses, labels)
  assert result['correct_answers'] == 3
  assert result['incorrect_answers'] == 1
  assert result['accuracy'] == 0.75
  cm = result['confusion_matrix']
  # Rows are correct answers, columns are responses.
  assert cm[0, 0] == 1  # 'a' heard as 'a'.
  assert cm[0, 1] == 1  # 'a' heard as 'b'.
  assert cm[1, 1] == 1  # 'b' heard as 'b'.
  assert cm[2, 2] == 1  # 'c' heard as 'c'.
  assert cm.sum() == 4


def test_analyze_results_empty():
  '''Empty response list returns zero counts and a zero matrix.'''
  result = analytics.analyze_results([], ['a', 'b'])
  assert result['correct_answers'] == 0
  assert result['incorrect_answers'] == 0
  assert result['accuracy'] == 0
  np.testing.assert_array_equal(
      result['confusion_matrix'], np.zeros((2, 2), dtype=int))


def test_create_confusion_matrix_shape_and_dtype():
  '''Confusion matrix has shape (n_labels, n_labels) and integer dtype.'''
  cm = analytics._create_confusion_matrix(
      [('a', 'a')], ['a', 'b', 'c', 'd'])
  assert cm.shape == (4, 4)
  assert np.issubdtype(cm.dtype, np.integer)
