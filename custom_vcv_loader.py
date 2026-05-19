"""ZIP-based custom VCV stimuli loader.

Extracts a user-provided ZIP file into a temporary directory, validates the
folder structure against the known VCV token set, checks WAV file constraints,
and returns a mapping from consonant labels to file paths.
"""

import os
import shutil
import tempfile
import zipfile

from scipy.io import wavfile

import bayesian_vcv_estimator

# --- Constants ---

MAX_WAV_FILES_PER_TOKEN = 10
MAX_WAV_DURATION_S = 5.0

# Comprehensive VCV-to-consonant-label lookup.
# Keys are lowercase VCV tokens; values are the button labels.
VCV_TO_LABEL = {
    'aba': 'B', 'ada': 'D', 'afa': 'F', 'aga': 'G',
    'aka': 'K', 'ala': 'L', 'ama': 'M', 'ana': 'N',
    'apa': 'P', 'ara': 'R', 'asa': 'S', 'asha': 'SH',
    'ata': 'T', 'atha': 'TH', 'ava': 'V', 'aza': 'Z',
    'adha': 'DH', 'azha': 'ZH', 'aja': 'J',
    'acha': 'CH', 'aha': 'H', 'awa': 'W', 'aya': 'Y',
}

# The set of consonant labels that are supported in the current test
# (i.e., those that have adaptive-mode priors and class assignments).
SUPPORTED_LABELS = set(bayesian_vcv_estimator.CONSONANT_LABELS.keys())


def extract_and_validate_zip(
    zip_bytes: bytes,
    dest_dir: str | None = None,
) -> tuple[dict[str, list[str]], list[str]]:
  """Extracts a VCV ZIP file and validates its contents.

  Expected ZIP structure::

      ABA/
          speaker1.wav
          speaker2.wav
      ADA/
          recording.wav
      ...

  Top-level folders are treated as VCV token names (case-insensitive).
  Only tokens that map to the current supported consonant set are kept.

  Args:
    zip_bytes: Raw bytes of the uploaded ZIP file.
    dest_dir: Directory to extract into. If None, a temporary directory
      is created. Caller is responsible for cleanup.

  Returns:
    A tuple of:
      - dict mapping consonant labels (e.g. 'B') to lists of extracted
        WAV file paths.
      - list of warning/info messages for the user.
  """
  warnings = []

  if dest_dir is None:
    dest_dir = tempfile.mkdtemp(prefix='vcv_custom_')

  # Extract ZIP.
  try:
    with zipfile.ZipFile(
        __import__('io').BytesIO(zip_bytes), 'r'
    ) as zf:
      zf.extractall(dest_dir)
  except zipfile.BadZipFile:
    warnings.append('❌ The uploaded file is not a valid ZIP archive.')
    return {}, warnings

  # Discover top-level folders.  Handle both flat structure
  # (ABA/, ADA/) and single-root-folder structure (MyStimuli/ABA/, ...).
  top_level_entries = [
      e for e in os.listdir(dest_dir)
      if os.path.isdir(os.path.join(dest_dir, e))
      and not e.startswith('__')  # skip __MACOSX etc.
      and not e.startswith('.')
  ]

  # Check if there's a single wrapper folder.
  actual_root = dest_dir
  if len(top_level_entries) == 1:
    candidate = os.path.join(dest_dir, top_level_entries[0])
    subentries = [
        e for e in os.listdir(candidate)
        if os.path.isdir(os.path.join(candidate, e))
        and not e.startswith('__')
        and not e.startswith('.')
    ]
    # If the single folder contains subfolders, treat it as a wrapper.
    if subentries:
      actual_root = candidate
      top_level_entries = subentries

  # Map folders to consonant labels.
  stimuli = {}  # {consonant_label: [file_paths]}

  for folder_name in sorted(top_level_entries):
    vcv_token = folder_name.lower().strip()
    label = VCV_TO_LABEL.get(vcv_token)

    if label is None:
      warnings.append(
          f'⚠️ Skipping unknown VCV folder: "{folder_name}"'
      )
      continue

    if label not in SUPPORTED_LABELS:
      warnings.append(
          f'⚠️ Skipping "{folder_name}" ({label}) — not in the '
          f'supported consonant set.'
      )
      continue

    folder_path = os.path.join(actual_root, folder_name)
    wav_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith('.wav') and not f.startswith('.')
    ])

    if not wav_files:
      warnings.append(
          f'⚠️ Folder "{folder_name}" contains no WAV files — skipping.'
      )
      continue

    if len(wav_files) > MAX_WAV_FILES_PER_TOKEN:
      warnings.append(
          f'⚠️ Folder "{folder_name}" has {len(wav_files)} WAVs; '
          f'using first {MAX_WAV_FILES_PER_TOKEN}.'
      )
      wav_files = wav_files[:MAX_WAV_FILES_PER_TOKEN]

    valid_paths = []
    for wav_name in wav_files:

      wav_path = os.path.join(folder_path, wav_name)

      # Validate duration.
      try:
        sr, data = wavfile.read(wav_path)
        duration_s = len(data) / sr
        if duration_s > MAX_WAV_DURATION_S:
          warnings.append(
              f'⚠️ "{folder_name}/{wav_name}" is {duration_s:.1f}s '
              f'(max {MAX_WAV_DURATION_S}s) — skipping.'
          )
          continue
      except Exception as e:  # pylint: disable=broad-exception-caught
        warnings.append(
            f'⚠️ Could not read "{folder_name}/{wav_name}": {e} '
            f'— skipping.'
        )
        continue

      valid_paths.append(wav_path)

    if valid_paths:
      stimuli[label] = valid_paths

  if not stimuli:
    warnings.append(
        '❌ No valid VCV stimuli found. Please check your ZIP structure.'
    )
  else:
    loaded = ', '.join(sorted(stimuli.keys()))
    warnings.insert(0, f'✅ Loaded consonants: {loaded}')

  return stimuli, warnings


def cleanup_temp_dir(path: str | None):
  """Safely removes a temporary directory."""
  if path and os.path.isdir(path):
    shutil.rmtree(path, ignore_errors=True)
