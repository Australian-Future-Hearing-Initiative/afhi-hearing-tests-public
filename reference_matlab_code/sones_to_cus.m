function cus = sones_to_cus_nh(sones, cu_factor, cu_exponent)

if nargin < 2
  cu_factor = 13.0;
end
if nargin < 3
  cu_exponent = 0.3;
end

cus = cu_factor*sones.^cu_exponent;
