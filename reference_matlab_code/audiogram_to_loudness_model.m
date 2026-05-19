function [loudness_model, smoothed_gram] = audiogram_to_loudness_model( ...
  audiogram, num_components)

component_coeffs = audiogram_to_component_coeffs(audiogram, num_components);

loudness_model = struct( ...
  'component_coeffs', component_coeffs, ...
  'sone_intersection', 24);  % (24 is about 86 dB SPL for "loud").


