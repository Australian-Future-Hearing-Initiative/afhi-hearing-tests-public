function dbspl = amplitude_to_dbspl(amplitude, FS_dB)

if nargin < 2
  FS_dB = 94;  % Arbitrarily calibrate to amplitude in pascals.
end

dbspl = 20*log10(amplitude) + FS_dB;
