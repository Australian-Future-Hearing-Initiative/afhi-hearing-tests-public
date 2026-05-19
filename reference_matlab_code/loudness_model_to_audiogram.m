function audiogram = loudness_model_to_audiogram(loudness_model)

frequencies = [250, 500, 1000, 1500, 2000, 3000, 4000, 6000, 8000]';

% Finding hearing_levels, difference in dB between the 1/16 sone normal
% curve and the 1/16 sone subject curve (1/16 sone is 0 phon) converted
% to normal sones, phons, and dbspl.
sones = 1/16;
sones_nh = sones_subject_to_sones_nh( ...
  sones, frequencies, loudness_model);
dbspl_nominal = phons_to_dbspl(frequencies, sones_to_phons(sones));
dbspl_actual = phons_to_dbspl(frequencies, sones_to_phons(sones_nh));
hearing_levels = dbspl_actual - dbspl_nominal;

audiogram = struct( ...
  'frequencies', frequencies, ...
  'hearing_levels', hearing_levels);
