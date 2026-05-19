function dbspl = phons_to_dbspl(freqs, phon_level, start_from_scratch, do_plots)
% Estimate constant loudness curve at specified frequencies, for one phon
% level, using polynomial function of frequency, linear db model.

if length(phon_level) ~= 1
  if length(phon_level) ~= length(freqs)
    error('In dbspl_for_phon, phon_level should be a scalar, or vector the same size as freqs.')
  end
end

if nargin < 3
  start_from_scratch = 0;  % normally just use save shortcut coefficients.
end

if nargin < 4
  do_plots = 0;  % normally silent
end

if start_from_scratch  % In case we want to update the algorith/data...
  do_model_plots = 1;  % Or use 0 to suppress the confirming plots.
  zero_and_rate_coeffs = phon_model_nh_coeffs(do_model_plots)
else
  zero_and_rate_coeffs = [ ...
    0.6725    1.0063
    1.1465    0.1293
    7.2111   -0.0947
    -123.9282   -0.0212
    25.3393   -0.1781
    246.3457    0.5013
    -50.2157   -0.0901
    -190.0229   -0.4636
    88.6428    0.2410
    % 0 1
    %
    % -2.6105    0.9715
    % -0.1030    0.0321
    % 8.4758   -0.0765
    % -23.9539    0.1376
    % 16.4674   -0.1031
    %
    % 0.3757    0.9842
    % -6.8525    0.0233
    % -17.9997   -0.1757
    % 3.2432    0.1824
    % 49.7382    0.0130
    % -27.4777   -0.0621
    %
    % 0.3715    0.9976
    % 7.5395    0.0473
    % -16.4027   -0.3987
    % -98.0869    0.1757
    % 68.0747    0.6542
    % 174.7954   -0.4342
    % -89.9373   -0.3878
    % -117.3947    0.4716
    % 84.2148   -0.1723
    % -11.1210    0.0199
    ];
end

order = size(zero_and_rate_coeffs, 1) - 1;
[audf, audf_powers] = CF_to_audf(freqs, order);
zero_and_rate_approx = audf_powers * zero_and_rate_coeffs;
zero_phon_curve = zero_and_rate_approx(:, 1);  % dB SPL nominal threshold
phon_rate_curve = zero_and_rate_approx(:, 2);  % dB per phon
dbspl = zero_phon_curve + phon_level.*phon_rate_curve;

if do_plots
  figure(1); clf
  semilogx(freqs, dbspl)
end

% Test:   dbspl_for_phon_nh(50:50:20000, 80, 0, 1)
