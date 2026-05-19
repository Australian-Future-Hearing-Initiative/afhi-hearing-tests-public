function [consistency_loss, outliers] = cls_model_consistency( ...
  frequencies, amplitudes, cus, loudness_model)

[consistency_loss, scores] = loudness_model_loss_function( ...
  frequencies, amplitudes, cus, loudness_model);

% Prune outliers and try again?

outliers = abs(scores) > 2;  % Sort of z_score but without mean removal.

[consistency_loss, scores] = loudness_model_loss_function( ...
  frequencies(~outliers), amplitudes(~outliers), cus(~outliers), loudness_model);
