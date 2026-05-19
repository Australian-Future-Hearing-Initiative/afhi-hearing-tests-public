function poly_coeffs = audiogram_poly_coeffs(num_components)
% Polynomial coeffcients for 4 principal components of hearing loss.
% as polynomial function on audf scale.
% num_component 1 to 4, defaulting to 4.
% These come from fitting a database of mild-moderate audiograms:

poly_coeffs = [ ...
   39.2009  -10.1437    4.3980   -2.1827
   27.1555   20.2498   11.6840    4.1437
   14.6875   33.1961  -21.5776   21.0738
  -10.8151  -27.4179  -15.3929  -14.7968
   -3.8048  -33.5089   16.4637  -23.0168
    3.3513   24.6996    0.4706   13.7262
    ];

if nargin == 1 && num_components >=1 && num_components <=4
  poly_coeffs = poly_coeffs(:, 1:num_components);
end
