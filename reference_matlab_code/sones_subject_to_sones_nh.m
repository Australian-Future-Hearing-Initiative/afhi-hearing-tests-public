function sones_nh = sones_subject_to_sones_nh( ...
  sones_subject, frequencies, subject_loudness_model)

[dbhl, slopes] = hearing_level_model(frequencies, ...
  subject_loudness_model.component_coeffs);
% This one is mostly about level calibration, a scalar, not
% frequency dependent; it can also handle some conductive loss.
sone_intersection = subject_loudness_model.sone_intersection;

% Then adjust for possibly higher-than-normal growth of loudness.
% For hi subject, slope > 1, the nh subject will sense a reduced range
% of level or loudness, so exponent < 1:
sones_nh = ((sones_subject/sone_intersection).^(1./slopes)) * ...
  sone_intersection;

% This power-law or "straight line in log space" model may be an
% over-simplification, but it avoids the complexities of more elaborate
% parameterizations that would be hard to get enough data for. 
