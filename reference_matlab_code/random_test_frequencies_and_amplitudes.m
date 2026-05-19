function [frequencies, amplitudes] = ...
  random_test_frequencies_and_amplitudes(min_freq, max_freq, ...
  how_many, loudness_model)
% Random test sample frequencies and amplitudes for the estimated
% loudness_model, which can be initialized like in update_loudness_model.

frequencies = exp(log(min_freq) + (log(max_freq) - log(min_freq))* ...
  rand(how_many, 1));

% Generate uniformly in phons or log(sones), from estimated middle of CU5
% (very soft) to estimated middle of CU45 (very loud), hoping to
% see about 5% in each of those extreme categories and about 11% in
% each of the 8 categories between them, and no "too loud".
% We'll likely ignore those extreme category responses, so they don't
% need to be balanced.

% Map top and bottom CUs for each frequency for the modeled subject:
min_cu = 5;
min_sones = sones_subject_to_sones_nh( ...
  cus_to_sones(min_cu), frequencies, loudness_model);
max_cu = 45;
max_sones = sones_subject_to_sones_nh( ...
  cus_to_sones(max_cu), frequencies, loudness_model);

% phons are defined as physical units, nh-related only.
min_phons = sones_to_phons(min_sones);
max_phons = sones_to_phons(max_sones);

% Generate uniform random levels in phon space.
phons = min_phons + (max_phons - min_phons) .* rand(how_many, 1);

% Convert to dbspl.
dbspl = phons_to_dbspl(frequencies, phons);

% Convert to amplitude.
% TODO(dicklyon): See if we need to calibrate a scale that's better.
dbfs = 94;  % Arbitrary full-scale scaling for now.
amplitudes = 10.^((dbspl-dbfs)/20);
% Use this amplitude as rms, for tone, warbled tone, or bandpass noise.

