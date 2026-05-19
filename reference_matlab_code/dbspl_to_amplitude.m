function amplitude = dbspl_to_amplitude(dbspl, FS_dB)

if nargin < 2
  FS_dB = 94;  % Arbitrarily calibrate to amplitude in pascals.
end

amplitude = 10.^((dbspl - FS_dB)/20);
