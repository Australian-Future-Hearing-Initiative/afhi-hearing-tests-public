"""Tests for custom_vcv_loader.py."""

import io
import tempfile
import zipfile

import numpy as np
from scipy.io import wavfile

import custom_vcv_loader


def _make_wav_bytes(duration_s: float = 1.0, sr: int = 44100) -> bytes:
  """Creates a minimal WAV file as bytes."""
  n_samples = int(sr * duration_s)
  data = np.zeros(n_samples, dtype=np.int16)
  buf = io.BytesIO()
  wavfile.write(buf, sr, data)
  return buf.getvalue()


def _make_zip_bytes(folder_contents: dict[str, list[tuple[str, bytes]]],
                    wrapper_folder: str | None = None) -> bytes:
  """Creates a ZIP file in memory.

  Args:
    folder_contents: {folder_name: [(filename, file_bytes), ...]}
    wrapper_folder: Optional wrapper folder name.

  Returns:
    ZIP file as bytes.
  """
  buf = io.BytesIO()
  with zipfile.ZipFile(buf, 'w') as zf:
    for folder, files in folder_contents.items():
      for fname, fbytes in files:
        if wrapper_folder:
          path = f'{wrapper_folder}/{folder}/{fname}'
        else:
          path = f'{folder}/{fname}'
        zf.writestr(path, fbytes)
  return buf.getvalue()


# ---------------------------------------------------------------------------
# VCV_TO_LABEL lookup
# ---------------------------------------------------------------------------

class TestVCVToLabel:
  """Tests for the VCV_TO_LABEL lookup table."""

  def test_standard_10_present(self):
    """All 10 standard consonants have entries."""
    expected = {'aba': 'B', 'ada': 'D', 'aga': 'G', 'aka': 'K',
                'ana': 'N', 'asa': 'S', 'asha': 'SH', 'ata': 'T',
                'ava': 'V', 'aza': 'Z'}
    for vcv, label in expected.items():
      assert custom_vcv_loader.VCV_TO_LABEL[vcv] == label

  def test_unknown_vcv_returns_none(self):
    """Unknown VCV tokens are not in the lookup."""
    assert custom_vcv_loader.VCV_TO_LABEL.get('axxa') is None


# ---------------------------------------------------------------------------
# extract_and_validate_zip
# ---------------------------------------------------------------------------

class TestExtractAndValidateZip:
  """Tests for extract_and_validate_zip()."""

  def test_valid_zip_two_consonants(self):
    """A well-formed ZIP with two VCV folders loads correctly."""
    wav = _make_wav_bytes(1.0)
    zip_bytes = _make_zip_bytes({
        'ABA': [('s1.wav', wav), ('s2.wav', wav)],
        'ADA': [('s1.wav', wav)],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
      stimuli, warnings = custom_vcv_loader.extract_and_validate_zip(
          zip_bytes, dest_dir=tmpdir
      )

    assert 'B' in stimuli
    assert 'D' in stimuli
    assert len(stimuli['B']) == 2
    assert len(stimuli['D']) == 1
    assert any('✅' in w for w in warnings)

  def test_case_insensitive_folders(self):
    """Folder names are matched case-insensitively."""
    wav = _make_wav_bytes(1.0)
    zip_bytes = _make_zip_bytes({
        'aba': [('s1.wav', wav)],
        'Aga': [('s1.wav', wav)],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
      stimuli, _ = custom_vcv_loader.extract_and_validate_zip(
          zip_bytes, dest_dir=tmpdir
      )

    assert 'B' in stimuli
    assert 'G' in stimuli

  def test_unknown_folder_skipped(self):
    """Folders with unrecognised VCV names are skipped with a warning."""
    wav = _make_wav_bytes(1.0)
    zip_bytes = _make_zip_bytes({
        'ABA': [('s1.wav', wav)],
        'UNKNOWN': [('s1.wav', wav)],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
      stimuli, warnings = custom_vcv_loader.extract_and_validate_zip(
          zip_bytes, dest_dir=tmpdir
      )

    assert 'B' in stimuli
    assert any('UNKNOWN' in w for w in warnings)

  def test_unsupported_consonant_skipped(self):
    """VCVs that map to consonants outside the supported set are skipped."""
    wav = _make_wav_bytes(1.0)
    # ALA -> L, which is not in our supported set.
    zip_bytes = _make_zip_bytes({
        'ABA': [('s1.wav', wav)],
        'ALA': [('s1.wav', wav)],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
      stimuli, warnings = custom_vcv_loader.extract_and_validate_zip(
          zip_bytes, dest_dir=tmpdir
      )

    assert 'B' in stimuli
    assert 'L' not in stimuli
    assert any('L' in w for w in warnings)

  def test_too_long_wav_skipped(self):
    """WAV files exceeding MAX_WAV_DURATION_S are skipped."""
    short_wav = _make_wav_bytes(1.0)
    long_wav = _make_wav_bytes(6.0)  # Over 5s limit.
    zip_bytes = _make_zip_bytes({
        'ABA': [('short.wav', short_wav), ('long.wav', long_wav)],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
      stimuli, warnings = custom_vcv_loader.extract_and_validate_zip(
          zip_bytes, dest_dir=tmpdir
      )

    assert len(stimuli['B']) == 1
    assert any('6.0s' in w for w in warnings)

  def test_empty_zip(self):
    """An empty ZIP returns no stimuli and an error message."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w'):
      pass

    with tempfile.TemporaryDirectory() as tmpdir:
      stimuli, warnings = custom_vcv_loader.extract_and_validate_zip(
          buf.getvalue(), dest_dir=tmpdir
      )

    assert not stimuli
    assert any('❌' in w for w in warnings)

  def test_wrapper_folder_handled(self):
    """A ZIP with a single root wrapper folder is unwrapped."""
    wav = _make_wav_bytes(1.0)
    zip_bytes = _make_zip_bytes(
        {'ABA': [('s1.wav', wav)]},
        wrapper_folder='MyStimuli'
    )

    with tempfile.TemporaryDirectory() as tmpdir:
      stimuli, _ = custom_vcv_loader.extract_and_validate_zip(
          zip_bytes, dest_dir=tmpdir
      )

    assert 'B' in stimuli

  def test_invalid_zip_data(self):
    """Invalid bytes produce an error, not a crash."""
    stimuli, warnings = custom_vcv_loader.extract_and_validate_zip(
        b'not a zip file'
    )
    assert not stimuli
    assert any('❌' in w for w in warnings)

  def test_empty_subfolder_skipped(self):
    """A subfolder with no WAVs is skipped with a warning."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
      # Create an empty directory entry.
      zf.writestr('ABA/', '')
      zf.writestr('ADA/s1.wav', _make_wav_bytes(1.0))

    with tempfile.TemporaryDirectory() as tmpdir:
      stimuli, warnings = custom_vcv_loader.extract_and_validate_zip(
          buf.getvalue(), dest_dir=tmpdir
      )

    assert 'B' not in stimuli
    assert 'D' in stimuli
    assert any('no WAV' in w for w in warnings)
