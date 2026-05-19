function [dbhl, slopes] = hearing_level_model(frequencies, model_coeffs)
% Return dB HL at the given frequencies, with hearing loss parameterized
% by a row of 1 to 4 coefficients.  First one is positive, 0 for NH.
% Subsequent coeffs are signed.  Mostly around -2 to 2, like z scores.
% [1, 0, 0] is the average mild-moderate hearing loss model.  Do grid
% search on coeffs with increment 0.1 or so to find a good fit.
% Recommend 3 coeffs for good enough personalization.
%
% This way of modeling dB HL can also be applied to other log-like
% measures of hearing loss, e.g. in phon space or log(sones) or log(CU),
% with appropriately scaled coefficients, without much inaccuracy. There
% the HL can be related to power-law exponents, or slopes.
% The slopes here represent faster-than-normal loudness growth, from
% threshold to 85 phon.

% These come from fitting a database of mild-moderate audiograms:
poly_coeffs = [ ...
   39.2009  -10.1437    4.3980   -2.1827
   27.1555   20.2498   11.6840    4.1437
   14.6875   33.1961  -21.5776   21.0738
  -10.8151  -27.4179  -15.3929  -14.7968
   -3.8048  -33.5089   16.4637  -23.0168
    3.3513   24.6996    0.4706   13.7262
    ];
poly_order = size(poly_coeffs, 1) - 1;
% Coeffs apply in the space of audf_powers.
[audf, audf_powers] = CF_to_audf(frequencies, poly_order);

num_components = length(model_coeffs);  % Must be 1 to 4
components = audf_powers * poly_coeffs(:, 1:num_components);

dbhl = components * model_coeffs';  % Prime assumes model_coeffs is a row.

slopes = dbhl_to_slopes(frequencies, dbhl);

% Test:  
% 
% f = 100:50:10000;
% figure; clf
% semilogx(f, hearing_loss_model(f, [1, 0, 0]))
% hold on
% semilogx(f, hearing_loss_model(f, [0, 0, 0]))
% semilogx(f, hearing_loss_model(f, [0.5, 0, 0]))
% semilogx(f, hearing_loss_model(f, [1, -1, 0]))
% semilogx(f, hearing_loss_model(f, [1, 1, 0]))
% semilogx(f, hearing_loss_model(f, [1, 0, -1]))
% semilogx(f, hearing_loss_model(f, [1, 0, 1]))
