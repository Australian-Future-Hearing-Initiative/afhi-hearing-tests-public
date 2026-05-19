function [loudness_model, rms_audiogram_error] = qCLS_testing_process( ...
  hidden_model, error_rate, batch_size, num_batches)

if nargin< 4
  num_batches = 5;
end
if nargin < 3
  batch_size = 20;
end
if nargin < 2
  error_rate = 1/3;
end
if nargin < 1
  hidden_coeffs = [1 0 0];  % mean mild-moderate
  hidden_model = struct( ...
    'component_coeffs', hidden_coeffs, ...  % Steep moderate loss.
    'sone_intersection', 24);
end

% Start by assuming the average mild-to-moderate loss:
loudness_model = struct( ...
  'component_coeffs', [1, 0], ...  % Average mild-moderate loss.
  'sone_intersection', 24);

% First batch on the most important range.
min_freq = 1000;
max_freq = 4000;

all_data = [];  % Or start with 0 rows, 3 columns.
cus = zeros(batch_size, 1);  % Collect responses here
for batch = 1:num_batches
  [frequencies, amplitudes] = ...
    random_test_frequencies_and_amplitudes(min_freq, max_freq, ...
    batch_size, loudness_model);

  % figure(1); clf
  % semilogx(frequencies, amplitude_to_dbspl(amplitudes), 'r*')

  % Run the test, get "cus" categorical data from user, ... 
  % ... needs some data to plug in here, replacing the randoms?
  for n = 1:batch_size
    cu = simulate_loudness_categorization( ...
      frequencies(n), amplitudes(n), hidden_model, error_rate);
    cus(n) = cu;
  end

  all_data = [all_data; [frequencies, amplitudes, cus]];

  % Then update the model part way, emphasizing the recent.
  learning_rate = 1.25 / (batch + 1);  % A crude update rate schedule, initially 0.75.

  [new_loudness_model, losses] = update_loudness_model( ...
    frequencies, amplitudes, cus, loudness_model, learning_rate);

  loudness_model = new_loudness_model

  disp(losses)  % Show loss functions updating

  % Expand the range for subsequent batches.
  min_freq = 250;
  max_freq = 8000;
end



% Now update based on all data, but still only partially to keep some
% emphasis on recent:
frequencies = all_data(:, 1);
amplitudes = all_data(:, 2);
cus = all_data(:, 3);


figure(1); clf
semilogx(frequencies, amplitude_to_dbspl(amplitudes), 'r*')
hold on
title('Simulated user response CU values')
xlabel('Frequency, Hz')
ylabel('dB SPL presented')
for n = 1:size(all_data, 1)
  text(frequencies(n), amplitude_to_dbspl(amplitudes(n)), num2str(cus(n))); ...
end

learning_rate = 0.5;
[loudness_model, losses] = update_loudness_model( ...
  frequencies, amplitudes, cus, loudness_model, learning_rate);
disp(losses);  % To see how much we lose by keeping rate low.

hidden_audiogram = loudness_model_to_audiogram(hidden_model)
inferred_audiogram = loudness_model_to_audiogram(loudness_model)

semilogx(hidden_audiogram.frequencies, hidden_audiogram.hearing_levels, 'bo-')
semilogx(inferred_audiogram.frequencies, inferred_audiogram.hearing_levels, 'rs-')

audigram_errors_db = (hidden_audiogram.hearing_levels - inferred_audiogram.hearing_levels)'
rms_audiogram_error = mean(audigram_errors_db.^2)^0.5

return

