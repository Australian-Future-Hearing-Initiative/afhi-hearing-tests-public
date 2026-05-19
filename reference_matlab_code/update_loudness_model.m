function [loudness_model, losses] = update_loudness_model( ...
  frequencies, amplitudes, cus, loudness_model, rate)
% Update the subject's loudness model based on categorical loudness test
% results, using correspond frequencies in Hz, amplitudes relative to
% full scale (i.e. the rms waveform value presented), cus being the 
% categorical units (CUs), from 0 to 50 by 5 per category.

if nargin < 4 || isempty(loudness_model)
  loudness_model = struct( ...
    'component_coeffs', [1, 0, 0], ...  % Average mild-moderate loss.
    'sone_intersection', 24);
end

old_loss = loudness_model_loss_function(frequencies, amplitudes, ...
  cus, loudness_model);

component_coeffs = loudness_model.component_coeffs;
sone_intersection = loudness_model.sone_intersection;

% Grid search on the component_coeffs to see if loss can be reduced.
% (this could/should all be done by gradient descent in Jax)
inc = 0.1;  % Arbitrary good-enough grid increment.
inc2 = 0.5;  % For the sone_intersection, near 24.
improved = 1;  % Just to make sure we go through at least once.
best_loss = old_loss;
while improved  % Iterate repeatedly over dimensions to tweak up or down.
  improved = 0;
  for dim = 1:length(component_coeffs)
    coeffs = component_coeffs;
    coeffs(dim) = component_coeffs(dim) + inc;
    test_model = struct('component_coeffs', coeffs, ...
      'sone_intersection', sone_intersection);
    new_loss = loudness_model_loss_function(frequencies, amplitudes, ...
      cus, test_model);
    if new_loss < best_loss
      best_loss = new_loss;
      improved = 1;
      component_coeffs = coeffs;
    end
    % Also try the negative inc, even if that one was good.
    coeffs(dim) = component_coeffs(dim) - inc;
    test_model = struct('component_coeffs', coeffs, ...
      'sone_intersection', sone_intersection);
    new_loss = loudness_model_loss_function(frequencies, amplitudes, ...
      cus, test_model);
    if new_loss < best_loss
      best_loss = new_loss;
      improved = 1;
      component_coeffs = coeffs;
    end
  end
  % Also search on the sone_intersection dimension here.
  % (could integrate this better as a dimension in the coeffs...)
  coeffs = component_coeffs;  % Make sure we use the best found so far.
  test_model = struct('component_coeffs', coeffs, ...
    'sone_intersection', sone_intersection + inc2);
  new_loss = loudness_model_loss_function(frequencies, amplitudes, ...
    cus, test_model);
  if new_loss < best_loss
    best_loss = new_loss;
    improved = 1;
    sone_intersection = sone_intersection + inc2;
  end
  test_model = struct('component_coeffs', coeffs, ...
    'sone_intersection', sone_intersection - inc2);
  new_loss = loudness_model_loss_function(frequencies, amplitudes, ...
    cus, test_model);
  if new_loss < best_loss
    best_loss = new_loss;
    improved = 1;
    sone_intersection = sone_intersection - inc2;
  end
end

% Update the model partially toward the new optimum, by an amount 
% 0 < rate <= 1:
loudness_model.component_coeffs = loudness_model.component_coeffs + ...
  rate * (component_coeffs - loudness_model.component_coeffs);
loudness_model.sone_intersection = loudness_model.sone_intersection + ...
  rate * (sone_intersection - loudness_model.sone_intersection);


component_coeffs = loudness_model.component_coeffs;
sone_intersection = loudness_model.sone_intersection;
updated_loss = loudness_model_loss_function(frequencies, amplitudes, ...
  cus, loudness_model);

losses = [old_loss, best_loss, updated_loss];
% Returns the updated loudness_model, along with losses.

