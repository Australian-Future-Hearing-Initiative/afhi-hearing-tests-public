function sones = cus_to_sones(cus, cu_factor, cu_exponent)

if nargin < 2
  cu_factor = 13;
end
if nargin < 3
  cu_exponent = 0.3;
end

sones = (cus/cu_factor).^(1/cu_exponent);
