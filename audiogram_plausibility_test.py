"""Tests for audiogram_plausibility."""

import csv
from pathlib import Path

import numpy as np
import pytest

import audiogram_data
import audiogram_plausibility as ap
import common

FREQS = [250, 500, 1000, 2000, 3000, 4000, 6000, 8000]
GOLDEN_PATH = (Path(__file__).resolve().parent / "tests" / "golden_data"
               / "audiogram_plausibility_cases.csv")
GOLDEN_TOLERANCE_DB = 1e-4


def _read_golden() -> tuple[dict[str, str], list[dict[str, str]]]:
  """Returns the golden file's 'NAME: value' header lines and its rows."""
  with open(GOLDEN_PATH, newline="", encoding="utf-8") as f:
    lines = f.read().splitlines()
  header = dict(line[2:].split(": ", 1) for line in lines
                if line.startswith("# ") and ": " in line[2:]
                and line[2:].split(": ", 1)[0].isupper())
  rows = list(csv.DictReader(line for line in lines
                             if not line.startswith("#")))
  return header, rows


_GOLDEN_HEADER, _GOLDEN_ROWS = _read_golden()


def test_golden_data_constants_match_module():
  # If this fails, the check was changed: regenerate the golden data with
  # generate_audiogram_plausibility_golden.py if the change is intended.
  assert float(_GOLDEN_HEADER["MAX_RMS_ERROR_DB"]) == ap.MAX_RMS_ERROR_DB
  assert float(_GOLDEN_HEADER["MIN_FREQUENCY_HZ"]) == ap.MIN_FREQUENCY_HZ
  assert float(_GOLDEN_HEADER["MAX_FREQUENCY_HZ"]) == ap.MAX_FREQUENCY_HZ
  assert float(_GOLDEN_HEADER["MIN_DBHL"]) == ap.MIN_DBHL
  assert float(_GOLDEN_HEADER["MAX_DBHL"]) == ap.MAX_DBHL
  assert tuple(int(f) for f in _GOLDEN_HEADER["REQUIRED_FREQUENCIES_HZ"]
               .split()) == ap.REQUIRED_FREQUENCIES_HZ
  assert int(_GOLDEN_HEADER["MIN_FREQUENCIES"]) == ap.MIN_FREQUENCIES


@pytest.mark.parametrize("row", _GOLDEN_ROWS,
                         ids=[r["case_id"] for r in _GOLDEN_ROWS])
def test_golden_case(row):
  audiogram = {int(k[3:]): float(v) for k, v in row.items()
               if k.startswith("hl_") and v}
  assert row["outcome"] in ("plausible", "implausible", "error")
  if row["outcome"] == "error":
    with pytest.raises(ValueError):
      ap.is_plausible(audiogram)
    return
  assert ap.is_plausible(audiogram) == (row["outcome"] == "plausible")
  if not row["rms_error_db"]:
    return
  rms_error_db = ap.fit_global_model(audiogram).rms_error_db
  assert rms_error_db == pytest.approx(float(row["rms_error_db"]),
                                       abs=GOLDEN_TOLERANCE_DB)


def test_recovers_model_audiogram():
  coeffs = np.array([1.0, 0.3, -0.2, 0.1])
  dbhl = ap.basis_shapes(np.array(FREQS, float)) @ coeffs
  fit = ap.fit_global_model(dict(zip(FREQS, dbhl)))
  np.testing.assert_allclose(fit.coeffs, coeffs, atol=1e-6)
  assert fit.rms_error_db < 1e-6


def test_is_plausible():
  dbhl = ap.basis_shapes(np.array(FREQS, float)) @ np.array([1.0, 0, 0, 0])
  assert ap.is_plausible(dict(zip(FREQS, dbhl)))
  zigzag = dbhl + 25.0 * (-1.0) ** np.arange(len(FREQS))
  assert not ap.is_plausible(dict(zip(FREQS, zigzag)))
  spike = dbhl.copy()
  spike[3] += 50.0
  assert not ap.is_plausible(dict(zip(FREQS, spike)))
  out_of_range = dbhl.copy()
  out_of_range[0] = 130.0
  assert not ap.is_plausible(dict(zip(FREQS, out_of_range)))


def test_bisgaard_audiograms_are_plausible():
  for library in (audiogram_data.BISGAARD_STANDARD_AUDIOGRAMS,
                  audiogram_data.BISGAARD_TYPICAL_AUDIOGRAMS):
    for name, audiogram in library.items():
      assert ap.is_plausible(audiogram), name


def test_is_plausible_rejects_frequencies_out_of_range():
  dbhl = dict(zip(FREQS, [15, 20, 25, 35, 45, 50, 55, 60]))
  for bad in (0, -100, 100, 10000, np.nan, np.inf):
    with pytest.raises(ValueError, match="Frequencies must be"):
      ap.is_plausible({**dbhl, bad: 40})
  assert ap.is_plausible({**dbhl, 125: 15})


def test_is_plausible_needs_required_frequencies():
  # Six frequencies, but 4000 Hz is missing.
  with pytest.raises(ValueError, match="4000"):
    ap.is_plausible({250: 15, 500: 20, 1000: 30, 2000: 40, 3000: 45,
                     6000: 55})


def test_is_plausible_needs_two_extra_frequencies():
  required = {500: 20, 1000: 30, 2000: 40, 4000: 50}
  with pytest.raises(ValueError, match="at least"):
    ap.is_plausible({**required, 3000: 45})
  assert ap.is_plausible({**required, 3000: 45, 6000: 55})


def test_non_finite_threshold_is_implausible():
  dbhl = dict(zip(FREQS, [15, 20, 25, 35, 45, 50, 55, 60]))
  for bad in (np.nan, np.inf, -np.inf):
    assert not ap.is_plausible({**dbhl, 2000: bad}), bad


def test_frequency_order_does_not_matter():
  # The app's STANDARD_FREQS_HZ is not in ascending order.
  dbhl = dict(zip(FREQS, [15, 20, 25, 35, 45, 50, 55, 60]))
  unsorted = {f: dbhl[f] for f in common.STANDARD_FREQS_HZ}
  assert list(unsorted) != sorted(unsorted)
  assert (ap.fit_global_model(unsorted).rms_error_db
          == ap.fit_global_model(dbhl).rms_error_db)


def test_single_frequency_spike_increases_max_error():
  dbhl = ap.basis_shapes(np.array(FREQS, float)) @ np.array([1.0, 0, 0, 0])
  smooth = ap.fit_global_model(dict(zip(FREQS, dbhl)))
  dbhl[3] += 40.0
  spiky = ap.fit_global_model(dict(zip(FREQS, dbhl)))
  assert spiky.max_abs_error_db > smooth.max_abs_error_db + 20.0
