function zero_and_rate_coeffs = phon_model_nh_coeffs(do_plots)

if nargin < 1
  do_plots = 1;
end

% Data from Robinson-Dadson 1956 paper.
% The f here is kHz; the a, b, and c apply to powers of dbspl to get phons
fabc = [ ...
  % 0.0200 -217.2000    3.6690   -0.0093
  0.0250 -167.2000    3.1450   -0.0078
  % 0.0300 -135.7000    2.8010   -0.0068
  0.0350 -113.2000    2.5380   -0.0059
  % 0.0400  -96.3000    2.3260   -0.0051
  0.0450  -83.2000    2.1550   -0.0045
  % 0.0500  -73.0000    2.0210   -0.0041
  0.0550  -65.1000    1.9170   -0.0037
  0.0600  -58.7000    1.8350   -0.0034
  0.0700  -49.1000    1.7220   -0.0031
  0.0800  -42.5000    1.6520   -0.0029
  0.0900  -37.4000    1.6030   -0.0027
  0.1000  -33.5000    1.5700   -0.0027
  0.1200  -27.0000    1.5120   -0.0026
  0.1400  -22.7000    1.4730   -0.0026
  0.1600  -19.4000    1.4440   -0.0025
  0.1800  -16.9000    1.4220   -0.0024
  0.2000  -14.7000    1.4040   -0.0024
  0.2500  -10.8000    1.3620   -0.0023
  0.3000   -8.1000    1.3250   -0.0022
  0.3500   -6.1000    1.2900   -0.0020
  0.4000   -4.7000    1.2590   -0.0019
  0.6000   -1.8000    1.1550   -0.0012
  0.8000   -0.5000    1.0640   -0.0005
  1.0000         0    1.0000         0
  1.5000    1.4000    0.9440    0.0006
  2.0000    3.3000    0.9240    0.0010
  2.5000    5.3000    0.9280    0.0012
  3.0000    6.9000    0.9370    0.0012
  3.5000    7.9000    0.9450    0.0011
  4.0000    7.9000    0.9540    0.0010
  4.5000    7.1000    0.9630    0.0008
  5.0000    5.3000    0.9730    0.0006
  6.0000   -0.5000    1.0110    0.0001
  7.0000   -7.5000    1.0750   -0.0003
  8.0000  -13.3000    1.1590   -0.0009
  9.0000  -16.5000    1.2420   -0.0014
  10.0000  -16.8000    1.3140   -0.0020
  11.0000  -14.8000    1.3770   -0.0027
  12.0000  -12.7000    1.4500   -0.0035
  13.0000  -13.9000    1.5660   -0.0045
  14.0000  -22.7000    1.7770   -0.0059
  15.0000  -43.0000    2.1460   -0.0077
  % 17.0000  -80.0000    2.2      -0.0060  % Fake hack
  % 20.0000  -200.000    3.0      -0.0050  % Fake hack
  % % 9.0000  -16.5000    1.2420   -0.0014
  % % 10.0000  -16.8000    1.3140   -0.0020
  % 11.0000  -14.8000    1.3770   -0.0027
  % % 12.0000  -12.7000    1.4500   -0.0035
  % % 13.0000  -13.9000    1.5660   -0.0045
  % % 14.0000  -22.7000    1.7770   -0.0059
  % % 15.0000  -43.0000    2.1460   -0.0077
  % % 17.0000  -80.0000    2.2      -0.0060  % Fake hack
  % % 20.0000  -200.000    3.0      -0.0050  % Fake hack
  ];

% Solve for a few phon levels and get their differences
% Use abc of quadratic formula, swapping a&c from Robinson-Dadson names.
a = fabc(:, 4);
b = fabc(:, 3);
c = fabc(:, 2);
freqs = 1000*fabc(:, 1);

poly_order = 4;
[f_aud, audf_powers] = CF_to_audf(freqs, poly_order);

% Compute dbspl for 2 or more selected phon levels (fit the first two).
phon_levels = [30 70];
iso_loudness_dbspl = zeros(length(f_aud), length(phon_levels));
for col = 1:length(phon_levels)
  ph = phon_levels(col);
  % First solve linear, pretending a is zero: 0 = b*x + c - ph
  dbspl_levels = -(c - ph)./b;
  % Then solve the quadratic where a is nonzero.
  nz = a ~= 0;
  dbspl_levels(nz) = (-b(nz) + (b(nz).^2 - 4*a(nz).*(c(nz) - ph)).^0.5) ./ (2*a(nz));
  iso_loudness_dbspl(:, col) = dbspl_levels;
end

% dB SPL from phons model is now in iso_loudness_dbspl and phon_levels.

dbspl_per_phon = diff(iso_loudness_dbspl')' ./ diff(phon_levels);
dbspl_at_0_phon = iso_loudness_dbspl(:,1) - phon_levels(1)*dbspl_per_phon;

if do_plots
  figure(15); clf
  semilogx(freqs, iso_loudness_dbspl)
  ylabel('dB SPL')
  xlabel('frequency, Hz')
  title('Robinson-Dadson iso-phon curves (solid) and models')
  hold on
  % Plot the linear model params; next fit poly to that.
  plot(freqs, dbspl_at_0_phon)
  plot(freqs, 100*dbspl_per_phon)
end

% Fit to selected points:
% Drop a few weird extreme-f points; makes it good enough for 100-15000 Hz.
which_points = find((mod(round(1000*fabc(:,1)),10)==0) & ...
  (fabc(:,1) ~= 9) & (fabc(:,1) ~= 10) & (fabc(:,1) ~= 12) & ...
  (fabc(:,1) ~= 13) & (fabc(:,1) ~= 10) & (fabc(:,1) ~= 17) & ...
  (fabc(:,1) ~= 8) & (fabc(:,1) ~= 14) ...
  );
rate_coeffs = audf_powers(which_points,:) \ dbspl_per_phon(which_points, 1);
zero_coeffs = audf_powers(which_points,:) \ dbspl_at_0_phon(which_points, 1);
zero_and_rate_coeffs = [zero_coeffs, rate_coeffs];

if do_plots
  % This is how the result is used to create the sbspl for various
  % frequencies and phons (zero_approx + rate_approx * phon_levels):
  rate_approx = audf_powers * rate_coeffs;
  zero_approx = audf_powers * zero_coeffs;
  plot(freqs, 100*rate_approx, ':', 'linewidth', 3)
  plot(freqs, zero_approx, ':', 'linewidth', 3)
  plot(freqs, zero_approx + rate_approx * phon_levels, ':', 'linewidth', 3)
  plot(freqs, zero_approx, ':', 'linewidth', 2)
  plot(freqs, zero_approx + rate_approx * phon_levels, ':', 'linewidth', 3)

  % % Compare with saved quanitized values, which is all we ever need from
  % % this function:
  % saved_zero_and_rate_coeffs = [ ...]
  %   0.3715    0.9976
  %   7.5395    0.0473
  %   -16.4027   -0.3987
  %   -98.0869    0.1757
  %   68.0747    0.6542
  %   174.7954   -0.4342
  %   -89.9373   -0.3878
  %   -117.3947    0.4716
  %   84.2148   -0.1723
  %   -11.1210    0.0199
  %   ];
  % saved_zero_coeffs = saved_zero_and_rate_coeffs(:, 1);
  % saved_rate_coeffs = saved_zero_and_rate_coeffs(:, 2);
  % rate_approx = audf_powers * saved_rate_coeffs;
  % zero_approx = audf_powers * saved_zero_coeffs;
  % plot(freqs, 100*rate_approx, '--', 'linewidth', 2)
  % plot(freqs, zero_approx, '--', 'linewidth', 2)
  % Looks perfect!
end
