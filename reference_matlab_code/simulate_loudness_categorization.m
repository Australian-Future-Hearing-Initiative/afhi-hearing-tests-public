function cu = simulate_loudness_categorization( ...
  frequency, amplitude, loudness_model, error_rate)
% Pick a category response for a presented tone frequency and amplitude,
% based on a subject's loudness model, with some errors.

% Reference data for category levels:
cus = [2, 5:5:50]';  % The "can't hear category must still be positive.
sones_subject = cus_to_sones(cus);

% Modified from the impaired model to normal:
sones_nh = sones_subject_to_sones_nh(sones_subject, frequency, loudness_model);
phons = sones_to_phons(sones_nh);
frequencies = frequency*ones(length(cus), 1);
dbspl = phons_to_dbspl(frequencies, phons);

presented_dbspl = amplitude_to_dbspl(amplitude);
% Now find distances to presented dbspl:
distances = (dbspl - presented_dbspl).^2;
% And pick the category with minimum distance, or not far from it.
[min_dist, index] = min(distances);
for it = 1:2  % To choose 1st, 2nd, or 3rd best category.
  % Make a chance of not picking the lowest distance, several times
  if rand(1) > error_rate
    break  % Stop looping and keep the current best.
  end
  distances(index) = 10000;
  [min_dist, index] = min(distances);
end
cu = cus(index);  % Return in category units, CU
