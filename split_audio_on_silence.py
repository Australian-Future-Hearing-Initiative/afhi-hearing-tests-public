"""
A command-line tool to split a single WAV file into multiple files based on
silent periods.

This script performs a series of operations:
1. Reads a list of labels from a text file (one label per line).
2. Splits a long audio file into individual utterances based on silence.
3. Saves the utterances as individual WAV files with format:
   <index>_<label>.wav (e.g., 01_hello.wav).

Usage:
python split_audio_on_silence.py --input <wav_file> --labels <labels_file> \
    --output <output_dir>
"""

import os
import argparse
from pydub import AudioSegment
from pydub.silence import split_on_silence


def load_labels(labels_file: str) -> list[str]:
  """
  Loads labels from a text file. Supports both comma-separated values and
  one label per line.

  Args:
      labels_file: Path to the text file containing labels.

  Returns:
      A list of label strings.
  """
  if not os.path.exists(labels_file):
    raise FileNotFoundError(f'Labels file not found: {labels_file}')

  with open(labels_file, 'r', encoding='utf-8') as f:
    content = f.read().strip()

  if not content:
    raise ValueError(f'No labels found in file: {labels_file}')

  # If content contains commas, treat as comma-separated.
  if ',' in content:
    labels = [label.strip() for label in content.split(',') if label.strip()]
  else:
    # Otherwise, treat as one label per line.
    labels = [line.strip() for line in content.split('\n') if line.strip()]

  if not labels:
    raise ValueError(f'No labels found in file: {labels_file}')

  return labels


def split_audio_on_silence(
    input_file: str,
    labels_file: str,
    output_dir: str,
    silence_thresh_db: float,
    min_silence_len_ms: int
):
  """
  Loads a WAV file, splits it on silence, and saves the individual chunks
  with numbered filenames.

  Args:
      input_file: Path to the source WAV file.
      labels_file: Path to a text file containing labels (one per line).
      output_dir: Directory to save the output files.
      silence_thresh_db: The upper bound for what is considered silence in dBFS
                         for the main splitting operation.
      min_silence_len_ms: The minimum length of a silence in ms to be used
                         for splitting.
  """
  # Load labels from file.
  labels = load_labels(labels_file)
  print(f'Loaded {len(labels)} labels from: {labels_file}')

  if not os.path.exists(input_file):
    raise FileNotFoundError(f'Input file not found: {input_file}')

  print(f'Loading audio file: {input_file}')
  audio = AudioSegment.from_wav(input_file)

  print(
      f'Splitting audio on silence (threshold: {silence_thresh_db} dBFS, '
      f'min length: {min_silence_len_ms} ms)...'
  )
  chunks = split_on_silence(
      audio,
      min_silence_len=min_silence_len_ms,
      silence_thresh=silence_thresh_db,
      keep_silence=300  # Keep a small amount of silence at edges.
  )

  # --- Validation ---
  if len(chunks) != len(labels):
    raise ValueError(
        f'Error: Expected {len(labels)} chunks based on the labels file, '
        f'but found {len(chunks)} audio chunks. '
        'Try adjusting the silence threshold or minimum silence length.'
    )
  print(f'Successfully split into {len(chunks)} chunks.')

  # --- Saving ---
  os.makedirs(output_dir, exist_ok=True)
  print(f'Saving chunks to directory: {output_dir}')

  # Determine zero-padding width for index based on number of labels.
  index_width = len(str(len(labels)))

  for i, chunk in enumerate(chunks):
    index_str = str(i + 1).zfill(index_width)
    output_filename = os.path.join(output_dir, f'{index_str}_{labels[i]}.wav')
    chunk.export(output_filename, format='wav')
    print(f'  Saved {output_filename}')

  print('\nProcessing complete.')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Split a WAV file into multiple files based on silence.'
  )
  parser.add_argument(
      '--input',
      type=str,
      required=True,
      help='Path to the input WAV file.'
  )
  parser.add_argument(
      '--labels',
      type=str,
      required=True,
      help='Path to a text file containing labels (one per line).'
  )
  parser.add_argument(
      '--output',
      type=str,
      required=True,
      help='Directory to save the split WAV files.'
  )
  parser.add_argument(
      '--silence-threshold',
      type=float,
      default=-40.0,
      help='The dBFS threshold for what is considered silence for splitting.'
  )
  parser.add_argument(
      '--min-silence-length',
      type=int,
      default=500,
      help='The minimum duration (ms) of silence between utterances.'
  )
  args = parser.parse_args()

  split_audio_on_silence(
      input_file=args.input,
      labels_file=args.labels,
      output_dir=args.output,
      silence_thresh_db=args.silence_threshold,
      min_silence_len_ms=args.min_silence_length
  )
