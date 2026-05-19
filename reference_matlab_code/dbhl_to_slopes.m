function slopes = dbhl_to_slopes(frequencies, dbhl)
% Leverage the parameterized model of HL to estimate loudness growth
% slopes relative to NH, at an arbitrary set of frequencies

db_threshold0 = phons_to_dbspl(frequencies, 0);
db_threshold_hi = dbhl + db_threshold0;
% Assume NH and HI converge linearized to loud.
loud_sones = 24;  % Nominal loud point, 86 phons, 24 sones.
top_phons = sones_to_phons(loud_sones);
db_loud = phons_to_dbspl(frequencies, top_phons);
inverse_slopes = (db_loud - db_threshold_hi) ./ (db_loud - db_threshold0);
% Take care of unexpected extreme and negative cases.
slopes = 1.0 ./ max(0.2, inverse_slopes);  % From near 1 up to 5 (beyond severe).
