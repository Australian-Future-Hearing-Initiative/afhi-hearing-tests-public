function [audf, audf_powers] = CF_to_audf(CF_Hz, order)
% An auditory-like frequency scale offset to 0 at 1 kHz, 
% ranging within -1 to +1.7 to cover 20 Hz to 20 kHz.
% And optionally powers of these to be used in polynomial models.

% Start with CAM scale parameters from
% Chen, Zhangli, Guangshu Hu, Brian R. Glasberg, and Brian CJ Moore. 
% "A new method of calculating auditory excitation patterns and loudness 
% for steady sounds." Hearing research 282, no. 1-2 (2011): 204-215.
break_f = 228.8;  % 1/0.00437
high_q = 9.294;  % 21.4/log(10)
cams = high_q * log(CF_Hz / break_f + 1);  % with sensible parameters.
cam1000 = high_q * log(1000 / break_f + 1);
% Normalize to a reasonable signed range for polynomial fitting.
audf = cams / cam1000 - 1;

if nargout > 1  % Then compute and return matrix of powers of audf
  if nargin < 2
    order = 9;  % 9 works well in loudness vs frequency modeling.
  end
  audf_powers = zeros(length(audf), order + 1);
  audf_powers(:, 1) = 1;  % First column is 0 power; avoid 0^0.
  for exponent = 1:order
    col = exponent + 1;
    audf_powers(:, col) = audf.^exponent;
  end
end
