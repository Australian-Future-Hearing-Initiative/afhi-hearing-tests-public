function mbdrc_params = loudness_model_to_mbdrc_fit( ...
  loudness_model, gain_limit_db, compression_rato_limit, fig_no)

if nargin < 4
  fig_no = 10;  % 0 to suppress plotting.
end
if nargin < 3
  compression_rato_limit = 2.5;
end
if nargin < 2
  gain_limit_db = 26;
end

[audiogram, four_frequency_average_hl] = loudness_model_to_audiogram(loudness_model);

% Maybe parameterize these limits, and decide what to do about them:
if four_frequency_average_hl < 21
  disp('Warning in loudness_model_to_mbdrc_fit: PTA4 is below mild.')
end
if four_frequency_average_hl > 56
  disp('Warning in loudness_model_to_mbdrc_fit: PTA4 is above moderate.')
end

frequencies = audiogram.frequencies;
hearing_levels = audiogram.hearing_levels;

% Stabilize by assuming even normals have a tiny positive HL, otherwise
% the threshold calculation will go bad.
hearing_levels = max(5, hearing_levels);

phon_threshold = 0;  % Nominal NH threshold.
dbspl_nominal = phons_to_dbspl(frequencies, phon_threshold);
% Subject's threshold in dbspl:
dbspl_subject = dbspl_nominal + hearing_levels;

phon_top = sones_to_phons(loudness_model.sone_intersection) - 5;  % Reduce gain above this level.
dbspl_top = phons_to_dbspl(frequencies, phon_top);

% Use compression to expand subject's range to NH range:
subject_range = dbspl_top - dbspl_subject;
normal_range = dbspl_top - dbspl_nominal;
subject_range = max(20, subject_range);  % Clip negative and too-small.

% Compression needed to restore normal rate of loudness growth:
compression_ratios = normal_range ./ subject_range;
compression_ratios = min(compression_rato_limit, compression_ratios);
gain_slope = (1 - 1./compression_ratios);  % Really slope is negative this.

% Make sure we didn't make slope 0 or negative (expansion), because we
% need to divide by it.
if any(gain_slope <= 0)
  error('gain_slope bug in audiogram_to_mbdrc_fit.m')
end

% Project down from top to get candidate threshold where gain limits out.
% This can go to very negative dbspl when compression ratio is near 1.
compression_threshold = dbspl_top - gain_limit_db ./ gain_slope;
% Limit it to be not below 30 dB SPL, rough noise floor.
compression_threshold = max(30, compression_threshold);
% Project down from unity-gain level to get low-level gain.
linear_region_gain_db = (dbspl_top - compression_threshold) .* gain_slope;

mbdrc_params = struct( ...
  'frequencies', frequencies, ...
  'compression_ratios', compression_ratios, ...
  'compression_threshold', compression_threshold, ...
  'linear_region_gain', linear_region_gain_db);

plot_mbdrc_prescription(mbdrc_params, fig_no);

