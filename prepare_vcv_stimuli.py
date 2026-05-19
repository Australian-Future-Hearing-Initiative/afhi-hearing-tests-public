"""
A command-line tool to prepare Vowel-Consonant-Vowel (VCV) stimuli from a
single source WAV file.

This script performs a series of operations:
1. Splits a long audio file into individual VCV utterances based on silence.
2. Trims leading silence from each utterance for consistent timing.
3. Resamples each utterance to a target sample rate (e.g., 44.1 kHz).
4. Standardizes the duration of any internal silent gaps within each VCV.
5. Saves the prepared utterances as individual, named WAV files.

Usage:
python prepare_vcv_stimuli.py --input <path_to_input_wav> --output <output_dir>
"""

import os
import argparse
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent

# The ordered list of VCVs expected in the input WAV file. This is used for
# naming the output files.
VCV_LABELS = [
    'aba', 'ada', 'afa', 'aga', 'aka', 'ana', 'asa', 'asha',
    'adha', 'ata', 'atha', 'ava', 'aza'
]


def prepare_vcv_stimuli(
    input_file: str,
    output_dir: str,
    silence_thresh_db: float,
    min_silence_len_ms: int,
    gap_thresh_db: float,
    min_gap_len_ms: int,
    standardized_gap_ms: int,
    target_sample_rate: int
):
  """
  Loads a WAV file, splits it, standardizes internal gaps, resamples, and
  saves the individual VCV chunks.

  Args:
      input_file: Path to the source WAV file.
      output_dir: Directory to save the output files.
      silence_thresh_db: The upper bound for what is considered silence in dBFS
                         for the main splitting operation.
      min_silence_len_ms: The minimum length of a silence in ms to be used
                         for splitting.
      gap_thresh_db: The dBFS threshold for detecting unwanted silences
                     (gaps) within each VCV chunk.
      min_gap_len_ms: The minimum duration (ms) of a silence to be
                      considered a gap for standardization.
      standardized_gap_ms: The target duration (in ms) for internal gaps.
      target_sample_rate: The sample rate (in Hz) to resample each chunk to.
  """
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
      keep_silence=100  # Keep a small amount of silence for trimming
  )

  # --- Validation ---
  if len(chunks) != len(VCV_LABELS):
    raise ValueError(
        f'Error: Expected {len(VCV_LABELS)} VCVs based on the labels, '
        f'but found {len(chunks)} audio chunks. '
        'Try adjusting the silence threshold or minimum silence length.'
    )
  print(f'Successfully split into {len(chunks)} chunks.')

  # --- Processing and Saving ---
  os.makedirs(output_dir, exist_ok=True)
  print(
      f'Processing and saving VCVs to directory: {output_dir}'
  )

  for i, chunk in enumerate(chunks):
    # 1. Standardize internal gaps by detecting audible parts and
    #    stitching them together with a fixed-duration silence.
    nonsilent_parts = detect_nonsilent(
        chunk,
        min_silence_len=min_gap_len_ms,
        silence_thresh=gap_thresh_db
    )

    if not nonsilent_parts:
      print(f'  Warning: Chunk {i+1} ({VCV_LABELS[i]}) appears to be '
            'completely silent. Skipping.')
      continue

    # Reconstruct the chunk with standardized gaps
    processed_chunk = AudioSegment.empty()
    standard_gap = AudioSegment.silent(duration=standardized_gap_ms)

    for j, (start_ms, end_ms) in enumerate(nonsilent_parts):
      processed_chunk += chunk[start_ms:end_ms]
      if j < len(nonsilent_parts) - 1:  # If not the last part
        processed_chunk += standard_gap

    # 2. Resample to the target rate
    resampled_chunk = processed_chunk.set_frame_rate(target_sample_rate)

    # 3. Save the final processed chunk
    output_filename = os.path.join(output_dir, f'{VCV_LABELS[i]}.wav')
    resampled_chunk.export(output_filename, format='wav')
    print(f'  Saved {output_filename}')

  print('\nProcessing complete.')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Prepare VCV stimuli from a single WAV file.'
  )
  parser.add_argument(
      '--input',
      type=str,
      required=True,
      help='Path to the input WAV file.'
  )
  parser.add_argument(
      '--output',
      type=str,
      required=True,
      help='Directory to save the prepared WAV files.'
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
      help='The minimum duration (ms) of silence between VCVs.'
  )
  parser.add_argument(
      '--gap-threshold',
      type=float,
      default=-55.0,
      help='The dBFS threshold for detecting internal gaps within each VCV.'
  )
  parser.add_argument(
      '--min-gap-length',
      type=int,
      default=100,
      help='The minimum duration (ms) of a silence to be considered a '
           'removable gap.'
  )
  parser.add_argument(
      '--standardized-gap-duration',
      type=int,
      default=100,
      help='The target duration (ms) for internal gaps after processing.'
  )
  parser.add_argument(
      '--sample-rate',
      type=int,
      default=44100,
      help='The target sample rate (Hz) to resample the audio to.'
  )
  args = parser.parse_args()

  prepare_vcv_stimuli(
      input_file=args.input,
      output_dir=args.output,
      silence_thresh_db=args.silence_threshold,
      min_silence_len_ms=args.min_silence_length,
      gap_thresh_db=args.gap_threshold,
      min_gap_len_ms=args.min_gap_length,
      standardized_gap_ms=args.standardized_gap_duration,
      target_sample_rate=args.sample_rate
  )
