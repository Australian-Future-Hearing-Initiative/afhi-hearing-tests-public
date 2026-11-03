"""Audiogram-based stimulus floor for the CLS test.

A PTA audiogram (as exported by the PTA test) can be loaded before a CLS test,
so that the CLS test does not spend trials on tones the subject cannot hear.
For each ear, the threshold is interpolated across log-frequency, converted to
dB SPL, and raised by a margin. Stimuli whose level falls below that floor are
redrawn rather than played, so the number of tones presented is unchanged.

This module is free of Streamlit calls so it can be unit tested directly.
"""

import io
import posixpath
import re
import zipfile
from typing import Callable

import numpy as np
import pandas as pd

import calibration

# Tones are only played if they are at least this far above threshold. PTA
# thresholds are roughly 50% detection points, so a small margin moves tones
# into the reliably audible range and absorbs calibration differences.
AUDIOGRAM_MARGIN_DB = 5.0
# dB HL is method-independent: each PTA algorithm has its own calibration
# table, so an audiogram means the same thing whichever algorithm produced it.
# Converting back to dB SPL for CLS playback therefore needs one hardware
# reference, used for every audiogram. The adaptive table is used because it
# estimates the 50% detection point and agrees with the pip table to within
# ~2 dB; the Hughson-Westlake table sits ~5 dB higher due to the procedural
# bias of its ascending 2-of-3 rule.
FLOOR_REFERENCE_CONFIGURATION = 'PixelBuds_Adaptive'
# Frequencies at which calibration.dbhl_to_dbspl has reference values.
_REFERENCE_FREQS_HZ = (250, 500, 1000, 2000, 3000, 4000, 6000, 8000)
# Maximum number of redraws before giving up and accepting a stimulus anyway.
MAX_REDRAW_ATTEMPTS = 1000
EARS = ('left', 'right')

# Per-ear audiogram: (frequencies in Hz, thresholds in dB HL), sorted by freq.
Audiogram = dict[str, tuple[np.ndarray, np.ndarray]]

_EAR_HEADER_PATTERN = re.compile(r'#\s*(left|right)-ear audiogram',
                                 re.IGNORECASE)


def parse_audiogram_csv(
    text: str, filename: str = ''
) -> tuple[str, np.ndarray, np.ndarray]:
  """Parses a single-ear audiogram CSV as written by the PTA test.

  The ear is read from the '# Left-Ear Audiogram' style header, falling back
  to 'left' or 'right' appearing in the file name.

  Args:
    text: The CSV file contents.
    filename: The file name, used only to infer the ear if needed.

  Returns:
    A tuple of (ear, frequencies in Hz, thresholds in dB HL), sorted by
    frequency, with any non-finite thresholds removed.

  Raises:
    ValueError: If the ear cannot be determined or no valid data is found.
  """
  match = _EAR_HEADER_PATTERN.search(text)
  if match:
    ear = match.group(1).lower()
  else:
    name = filename.lower()
    ears_in_name = [e for e in EARS if e in name]
    if len(ears_in_name) != 1:
      raise ValueError(f'Could not determine the ear for "{filename}".')
    ear = ears_in_name[0]
  try:
    df = pd.read_csv(io.StringIO(text), comment='#', header=None)
  except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
    raise ValueError(f'Could not read audiogram "{filename}": {e}') from e
  # Require exactly (frequency, threshold) columns, so another CSV uploaded
  # by mistake is rejected rather than partially read.
  if df.shape[1] != 2:
    raise ValueError(f'"{filename}" does not look like an audiogram: expected '
                     f'2 columns, found {df.shape[1]}.')
  freqs = pd.to_numeric(df[0], errors='coerce').to_numpy(float)
  hls = pd.to_numeric(df[1], errors='coerce').to_numpy(float)
  valid = np.isfinite(freqs) & np.isfinite(hls) & (freqs > 0)
  if not np.any(valid):
    raise ValueError(f'No valid thresholds found in "{filename}".')
  order = np.argsort(freqs[valid])
  return ear, freqs[valid][order], hls[valid][order]


def _is_metadata_entry(member: str) -> bool:
  """True for macOS zip metadata (e.g. __MACOSX/._name) and hidden files."""
  return (member.startswith('__MACOSX/')
          or posixpath.basename(member).startswith('.'))


def load_audiogram_files(files: list[tuple[str, bytes]]) -> Audiogram:
  """Loads per-ear audiograms from uploaded CSV and/or PTA results zip files.

  Args:
    files: List of (file name, file contents) pairs. Zip files are searched
      for members whose names end in 'audiogram.csv'.

  Returns:
    Dict mapping 'left' and/or 'right' to (frequencies, thresholds).

  Raises:
    ValueError: If no audiogram is found, a file cannot be read, or the same
      ear is supplied more than once.
  """
  csv_files = []
  for name, content in files:
    if name.lower().endswith('.zip'):
      try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
          for member in zf.namelist():
            if (member.lower().endswith('audiogram.csv')
                and not _is_metadata_entry(member)):
              csv_files.append((member, zf.read(member)))
      except zipfile.BadZipFile as e:
        raise ValueError(f'"{name}" is not a valid zip file.') from e
    else:
      csv_files.append((name, content))
  audiogram = {}
  for name, content in csv_files:
    try:
      text = content.decode('utf-8-sig')
    except UnicodeDecodeError as e:
      raise ValueError(f'"{name}" is not a text CSV file.') from e
    ear, freqs, hls = parse_audiogram_csv(text, name)
    if ear in audiogram:
      raise ValueError(f'More than one {ear}-ear audiogram was provided.')
    audiogram[ear] = (freqs, hls)
  if not audiogram:
    raise ValueError('No audiogram found in the uploaded file(s).')
  return audiogram


def threshold_dbspl(frequency_hz: float, freqs_hz: np.ndarray,
                    hls: np.ndarray) -> float:
  """Estimates the threshold in dB SPL at an arbitrary frequency.

  Both the dB HL thresholds and the dB HL to dB SPL reference are linearly
  interpolated in log-frequency (and held constant beyond their end points).
  """
  log_f = np.log(frequency_hz)
  hl = np.interp(log_f, np.log(freqs_hz), hls)
  ref_offsets = [
      calibration.dbhl_to_dbspl(0, f, FLOOR_REFERENCE_CONFIGURATION)
      for f in _REFERENCE_FREQS_HZ
  ]
  ref = np.interp(log_f, np.log(_REFERENCE_FREQS_HZ), ref_offsets)
  return float(hl + ref)


def floor_dbspl(frequency_hz: float, ear: str,
                audiogram: Audiogram) -> float | None:
  """Returns the lowest level (dB SPL) at which a tone may be played.

  For binaural ('both') stimuli the better ear is used, so the tone is
  audible to at least one ear. Returns None if there is no audiogram for the
  relevant ear(s), meaning no floor applies.
  """
  ears = EARS if ear == 'both' else (ear,)
  thresholds = [threshold_dbspl(frequency_hz, *audiogram[e])
                for e in ears if e in audiogram]
  if not thresholds:
    return None
  return min(thresholds) + AUDIOGRAM_MARGIN_DB


def floor_amplitude(frequency_hz: float, ear: str,
                    audiogram: Audiogram) -> float | None:
  """As floor_dbspl, but returns a linear playback amplitude."""
  db_spl = floor_dbspl(frequency_hz, ear, audiogram)
  return None if db_spl is None else calibration.dbspl_to_amp(db_spl)


def draw_audible_stimulus(
    draw_fn: Callable[[], tuple[float, float]],
    floor_fn: Callable[[float], float | None],
    max_amplitude: float,
    max_attempts: int = MAX_REDRAW_ATTEMPTS,
) -> tuple[float, float]:
  """Draws stimuli until one is above the audibility floor.

  A stimulus is accepted if its amplitude, after capping at max_amplitude, is
  at or above the floor for its frequency. Frequencies whose floor exceeds the
  cap are therefore skipped. If nothing acceptable is found within
  max_attempts, the last draw is returned so the test can continue.

  Args:
    draw_fn: Returns a (frequency in Hz, amplitude) candidate.
    floor_fn: Returns the floor amplitude for a frequency, or None.
    max_amplitude: The current maximum allowed amplitude.
    max_attempts: Maximum number of candidates to draw.

  Returns:
    The accepted (frequency in Hz, amplitude). The amplitude is not capped.
  """
  for _ in range(max_attempts):
    frequency_hz, amplitude = draw_fn()
    floor = floor_fn(frequency_hz)
    if floor is None or min(amplitude, max_amplitude) >= floor:
      return frequency_hz, amplitude
  print(f'Warning: no stimulus above the audiogram floor after '
        f'{max_attempts} attempts; playing the last candidate.')
  return frequency_hz, amplitude


def generate_floor_csv(audiogram: Audiogram) -> str:
  """Generates a CSV describing the audiogram and floor used in a CLS test."""
  rows = []
  for ear in EARS:
    if ear not in audiogram:
      continue
    for freq, hl in zip(*audiogram[ear]):
      rows.append({
          'Ear': ear,
          'Frequency (Hz)': freq,
          'Threshold (dB HL)': hl,
          'Floor (dB SPL)': round(floor_dbspl(freq, ear, audiogram), 1),
      })
  buffer = io.StringIO()
  buffer.write('# Audiogram used to set the CLS stimulus floor\n')
  buffer.write(f'# Margin above threshold: {AUDIOGRAM_MARGIN_DB} dB\n')
  buffer.write(f'# dB HL to dB SPL reference: '
               f'{FLOOR_REFERENCE_CONFIGURATION}\n')
  pd.DataFrame(rows).to_csv(buffer, index=False)
  return buffer.getvalue()
