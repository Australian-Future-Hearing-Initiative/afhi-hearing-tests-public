""" Tool for converting stereo WAV files to mono (left or right channel). """

import numpy as np
import scipy.io.wavfile as wavfile
import os
import argparse
import glob


def convert_stereo_to_mono_wav(input_wav_path: str, ear_to_keep: str):
  """
  Reads a stereo WAV file, zeros out one channel, and returns modified data.

  Handles mono input by duplicating the channel before zeroing.

  Args:
      input_wav_path: Path to the input WAV file.
      ear_to_keep: The ear/channel to keep ('left' or 'right'). The other
                   channel will be zeroed out.

  Raises:
      ValueError: If ear_to_keep is invalid, or audio shape is unexpected.
      FileNotFoundError: If input_wav_path does not exist.
      IOError, OSError: If file reading fails.

  Returns:
      tuple: (sample_rate, modified_audio_data) where modified_audio_data
             is a stereo NumPy array with one channel zeroed.
  """
  if ear_to_keep not in ['left', 'right']:
    raise ValueError("ear_to_keep must be 'left' or 'right'.")

  if not os.path.exists(input_wav_path):
    raise FileNotFoundError(f'Input audio file not found: {input_wav_path}')

  try:
    fs, audio_data = wavfile.read(input_wav_path)

    # Ensure data is stereo (handle mono by duplication).
    if audio_data.ndim != 2 or audio_data.shape[1] != 2:
      if audio_data.ndim == 1:
        print(
            f'Warning: Input {input_wav_path} is mono. '
            f'Duplicating channel before zeroing.'
        )
        audio_data = np.repeat(audio_data[:, np.newaxis], 2, axis=1)
      else:
        raise ValueError(
            f'Audio file {input_wav_path} has unexpected shape: '
            f'{audio_data.shape}'
        )

    # Create a writable copy.
    modified_data = audio_data.copy()

    # Zero out the appropriate channel.
    if ear_to_keep == 'left':
      modified_data[:, 1] = 0  # Zero out right channel.
    else: # ear_to_keep == 'right'
      modified_data[:, 0] = 0  # Zero out left channel.

    return fs, modified_data

  except (ValueError, IOError, OSError) as e:
    print(
        f'Error processing audio file {input_wav_path} '
        f'to keep {ear_to_keep} ear: {e}'
    )
    # Re-raise to signal failure to the main script.
    raise


def main():
  """Main function to parse arguments and perform conversion."""
  parser = argparse.ArgumentParser(
      description='Convert stereo WAV files to mono (L/R channel only) WAV.'
  )
  parser.add_argument(
      '-i', '--input', required=True,
      help='Path to a single input stereo WAV file or directory of WAV files.'
  )
  parser.add_argument(
      '-o', '--output-dir', required=True,
      help='Directory to save the converted mono WAV files.'
  )
  parser.add_argument(
      '-e', '--ear', required=True, choices=['left', 'right'],
      help="The ear/channel to KEEP ('left' or 'right')."
  )
  parser.add_argument(
      '--suffix', default=None,
      help="Suffix to add before the file extension (e.g., '_left'). "
           "Defaults to '_left' or '_right' based on --ear."
  )

  args = parser.parse_args()

  # Determine input files
  input_path = args.input
  if os.path.isdir(input_path):
    input_files = glob.glob(os.path.join(input_path, '*.wav'))
    if not input_files:
      print(f'No .wav files found in directory: {input_path}')
      return
  elif os.path.isfile(input_path) and input_path.lower().endswith('.wav'):
    input_files = [input_path]
  else:
    parser.error(
        f'Input path is not a valid WAV file or directory: {input_path}'
    )
    return # Should not be reached due to parser.error.

  # Create output directory if it doesn't exist
  output_dir = args.output_dir
  os.makedirs(output_dir, exist_ok=True)

  # Determine suffix
  suffix = args.suffix if args.suffix is not None else f'_{args.ear}'

  # Process files
  print(f'Starting conversion for ear: {args.ear}')
  print(f'Input source: {input_path}')
  print(f'Output directory: {output_dir}')
  print(f'Output suffix: {suffix}.wav')

  processed_count = 0
  error_count = 0
  for infile in input_files:
    try:
      print(f'Processing: {os.path.basename(infile)} ...')
      fs, modified_data = convert_stereo_to_mono_wav(infile, args.ear)

      # Construct output filename
      base_name = os.path.basename(infile)
      name_part, ext = os.path.splitext(base_name)
      output_filename = f'{name_part}{suffix}{ext}'
      output_path = os.path.join(output_dir, output_filename)

      # Write the modified data
      wavfile.write(output_path, fs, modified_data)
      processed_count += 1

    except (IOError, OSError, ValueError, FileNotFoundError) as e:
      # Catch specific errors during processing/writing for a single file.
      print(f'  ERROR processing {os.path.basename(infile)}: {e}')
      error_count += 1

  print('\nConversion complete.')
  print(f'  Successfully processed: {processed_count} files.')
  print(f'  Errors encountered: {error_count} files.')


if __name__ == '__main__':
  main()
