"""Functions for analysing the results of the consonant confusion test."""

import numpy as np

# Token stored when the listener reports that no speech was heard.
NO_SPEECH_RESPONSE = 'NO_SPEECH'


def _create_confusion_matrix(responses, label_list):
  """Creates a confusion matrix from the given responses.

  Args:
    responses: A list of tuples containing the response and correct answer.
    label_list: A list of all possible labels.

  Returns: A 2D numpy array representing the confusion matrix.
  """
  n_labels = len(label_list)
  confusion_matrix = np.zeros((n_labels, n_labels), dtype=int)
  for response, correct_answer in responses:
    if (response not in label_list or
        correct_answer not in label_list):
      # Skip no-speech and any other non-consonant response.
      continue
    if response == correct_answer:
      # Find index of response/correct answer in the label list.
      i = label_list.index(correct_answer)
      confusion_matrix[i, i] += 1  # Increment diagonal for correct answers.
    else:
      i = label_list.index(correct_answer)
      j = label_list.index(response)
      confusion_matrix[i, j] += 1  # Increment off-diagonal.
  return confusion_matrix

def analyze_results(responses, label_list):
  """Analyzes the test results and returns a dictionary of metrics."""
  correct_answers = 0
  incorrect_answers = 0
  no_speech_count = 0

  for response, correct_answer in responses:
    if response == NO_SPEECH_RESPONSE:
      no_speech_count += 1
    if response == correct_answer:
      correct_answers += 1
    else:
      incorrect_answers += 1

  accuracy = correct_answers / len(responses) if responses else 0
  return {
      'correct_answers': correct_answers,
      'incorrect_answers': incorrect_answers,
      'accuracy': accuracy,
      'no_speech_count': no_speech_count,
      'confusion_matrix': _create_confusion_matrix(responses, label_list)
  }
