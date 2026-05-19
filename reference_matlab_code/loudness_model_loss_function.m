function loss = loudness_model_loss_function( ...
  frequencies, amplitudes, cus, subject_loudness_model);
% Sum squared error between the dbspl levels corresponding to the
% categorical units, for a batch of test data.

OK = find((cus > 0) & (cus < 45));  % Skip "can't hear" and "very loud".
if isempty(OK)
  loss = 0;  % Because 0 can't be improved by changing the model.
else
  % Prune to just the ones in the range we want to fit:
  frequencies = frequencies(OK);
  amplitudes = amplitudes(OK);
  cus = cus(OK);
  sones_subject = cus_to_sones(cus);
  % Reduced range of NH sones expected for subject's modeled impairment:
  sones_nh = sones_subject_to_sones_nh( ...
    sones_subject, frequencies, subject_loudness_model);
  phons = sones_to_phons(sones_nh);
  % Not the presented dbspl, but the one with same loudness in NH:
  dbspl = phons_to_dbspl(frequencies, phons);
  % Presented levels:
  presented_dbspl = 20*log10(amplitudes) + 94;  % assuming a calibration to Pascals.
  loss = mean((dbspl - presented_dbspl).^2).^0.5;  % Error in rms dB
end
