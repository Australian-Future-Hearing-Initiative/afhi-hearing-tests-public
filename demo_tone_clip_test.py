"""Pulsed tones at full scale don't wrap around in the integer casts."""

from unittest.mock import patch

import numpy as np
import pytest

import demo_pip
import demo_pta

# A wrap turns a sample near +full scale into one near -full scale, a jump of
# about twice full scale. A 1 kHz tone moves at most ~15% of full scale per
# sample, and the noise below adds well under 1%, so half is a safe bound.
MAX_STEP_FRACTION = 0.5

# The noise the app adds: the cache-busting noise alone, and the tone
# generator's dither (DEFAULT_DITHER_SPL). Without the clip, every full-scale
# integer tone wrapped at both levels (200 of 200 seeds, 250 Hz to 4 kHz).
NOISE_LEVELS_DB_SPL = [None, 15.0]


def _played_tone(module, datatype, dither_db_spl, seed=0):
  """Plays a full-scale pulsed tone and returns the left channel."""
  np.random.seed(seed)
  with patch.object(module.common, 'autoplay_wav') as mock_play, \
       patch.object(module.st, 'warning'):
    module.play_pulsed_tone(1000, 1.0, datatype=datatype,
                            dither_db_spl=dither_db_spl)
  data, _ = mock_play.call_args[0]
  return data[:, 0]


@pytest.mark.parametrize('module', [demo_pta, demo_pip])
@pytest.mark.parametrize('dither_db_spl', NOISE_LEVELS_DB_SPL)
@pytest.mark.parametrize('datatype,full_scale', [
    ('int16', np.iinfo(np.int16).max),
    ('int32', np.iinfo(np.int32).max),
])
def test_full_scale_tone_does_not_wrap(module, dither_db_spl, datatype,
                                       full_scale):
  '''No sample jumps from one end of the range to the other.'''
  tone = _played_tone(module, datatype, dither_db_spl).astype(np.int64)
  assert np.max(np.abs(np.diff(tone))) < MAX_STEP_FRACTION * full_scale


@pytest.mark.parametrize('module', [demo_pta, demo_pip])
@pytest.mark.parametrize('dither_db_spl', NOISE_LEVELS_DB_SPL)
def test_full_scale_float_tone_stays_in_range(module, dither_db_spl):
  '''Float tones are kept within [-1, 1] too.'''
  tone = _played_tone(module, 'float32', dither_db_spl)
  assert np.max(np.abs(tone)) <= 1.0
