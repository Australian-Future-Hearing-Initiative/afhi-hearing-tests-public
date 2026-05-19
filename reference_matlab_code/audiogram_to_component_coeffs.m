function [component_coeffs, smoothed_gram] = audiogram_to_component_coeffs( ...
  audiogram, num_components)

frequencies_col = audiogram.frequencies;
frequencies_col = frequencies_col(:);  % Forces it to be a column vector.
hls_col = audiogram.hearing_levels;
hls_col = hls_col(frequencies_col <= 8000);
frequencies_col = frequencies_col(frequencies_col <= 8000);

% Repeat the 1, 2, 4 kHz key points to fit better.
for f = [1000, 2000, 4000]
  index = find(frequencies_col == f, 1);
  if ~isempty(index)
    frequencies_col = [frequencies_col; f];
    hls_col = [hls_col; hls_col(index)];
  end
end

poly_coeffs = audiogram_poly_coeffs(num_components);
order = size(poly_coeffs, 1) - 1;
[audf, audf_powers] = CF_to_audf(frequencies_col, order);
components = audf_powers * poly_coeffs;

component_coeffs = components \ hls_col;

if nargout > 1  % Return reconstructed smoothed audiogram
  frequencies_col = audiogram.frequencies;
  % Return smoothed audiogram at all frequencies
  [audf, audf_powers] = CF_to_audf(audiogram.frequencies, order);
  components = audf_powers * poly_coeffs;
  smoothed_hls = components * component_coeffs;
  smoothed_gram = struct( ...
    'frequencies', frequencies_col, ...
    'hearing_levels', smoothed_hls');  % Store as a row
end


