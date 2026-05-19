function [frequency, amplitude] = ...
  random_test_frequency_and_amplitude(min_freq, max_freq, ...
  loudness_model, prev_frequencies, prev_amplitudes)

% dbfs = 94;  % Arbitrary full-scale scaling for now.
min_cu = 3;
max_cu = 45;
full_phon_range = 90;  % Prune relative to this range.


OK = 0;
while ~OK  % Try multiple times to generate a "good" random pair.
  frequency = exp(log(min_freq) + (log(max_freq) - log(min_freq))* rand(1));
  % Map top and bottom CUs for this frequency for the modeled subject:
  min_sones = sones_subject_to_sones_nh( ...
    cus_to_sones(min_cu), frequency, loudness_model);
  max_sones = sones_subject_to_sones_nh( ...
    cus_to_sones(max_cu), frequency, loudness_model);
  % phons are defined as physical units, nh-related only.
  min_phons = sones_to_phons(min_sones);
  max_phons = sones_to_phons(max_sones);
  % Don't let the range get too high or too small.
  max_phons = min(max_phons, 90);
  min_phons = min(min_phons, max_phons - 20);  % Make extra inaudible highs.
  if rand(1) < (max_phons - min_phons)/full_phon_range  % Prune where range is small.
    OK = 1;
    phon_level = min_phons + (max_phons - min_phons) .* rand(1);
  end
end
amplitude = dbspl_to_amplitude(phons_to_dbspl(frequency, phon_level));
