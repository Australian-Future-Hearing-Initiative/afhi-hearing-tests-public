"""Functions for analyzing audiogram data."""

import numpy as np

def get_threshold(results: list[tuple[float, bool]],
                  min_level: float,
                  max_level: float) -> float | None:
  """Calculates the hearing threshold for data from one frequency.

  The threshold is defined here as the lowest level at which the tone was heard
  after a previous trial where a tone was not heard and where either a) two
  consistent levels were heard or b) three such levels were heard, in which
  case the middle level is taken as the threshold.

  Edge cases:
  1. If min_level is heard three times in a row, then it is the threshold.
  2. If max_level is not heard twice in a row, then it is the threshold.

  Args:
    results: A list of tuples, where each tuple represents a trial
             and contains (level, heard).
    min_level: The minimum possible level in dB HL.
    max_level: The maximum possible level in dB HL.
  Returns:
    The hearing threshold for the given frequency or None if it was not
    possible to calculate a threshold from the data.
  """
  # If there are no results, then we can't set a threshold.
  if not results:
    return None
  # Check edge case for min level heard three times in a row.
  if (len(results) >= 3 and
      all([level == min_level and
          heard for level, heard in results[-3:]])):
    return min_level
  # Check edge case for max level not heard twice in a row.
  last_two_trials = results[-2:]
  if all([level == max_level and
          not heard for level, heard in last_two_trials]):
    return max_level
  # Extract all heard tones that qualify towards the threshold calculation.
  qualifying_levels = []
  previous_tone_heard = True  # Assume the tone prior to the first was heard.
  for level, heard in results:
    if heard:
      if not previous_tone_heard:
        qualifying_levels.append(level)
      previous_tone_heard = True
    if not heard:
      previous_tone_heard = False
  # Now we have extracted all qualifying levels, so we can try to calculate the
  # threshold.
  if len(qualifying_levels) < 2:
    return None  # Insufficient data to calculate a threshold.
  elif len(qualifying_levels) == 2:
    # With only two qualifying levels, we need to continue to try to get a third
    # as a tie-breaker, unless they are identical. Strict equality is
    # ok here because levels are used in 5 dB steps.
    if int(qualifying_levels[0]) == int(qualifying_levels[1]):
      return qualifying_levels[0]
    else:
      return None
  else:
    # With three qualifying levels, the median is the threshold. This is used
    # because it's essentially a 'majority vote' between the three levels, but
    # in the case where all three are different, the middle level is chosen.
    return np.median(qualifying_levels)
