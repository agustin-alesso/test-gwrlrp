# Packages
# pkg install -forge mapping
pkg load mapping

# Read data
rosas = shaperead("rosas99")

# Inputs 
y = extractfield(rosas, "YIELD")
x = extractfield(rosas, "N")
east = extractfield(rosas, "X")
north = extractfield(rosas, "Y")

# Parameters
% info = a structure variable with fields:
% info.bwidth = scalar bandwidth to use or zero for cross-validation estimation (default)
pars = {}
pars.bwidth = 0
% info.bmin   = minimum bandwidth to use in CV search
pars.bmin = 0.1
% info.bmax   = maximum bandwidth to use in CV search
pars.bmax = 20
% info.dtype  = 'gaussian' (default), 'exponential', 'tricube'
pars.dtype  = 'gaussian'
% info.q      = q-nearest neighbors to use for tri-cube weights (default: CV estimated)
nvar = size(x,1)
pars.qmin   = nvar+2
pars.qmax   = 4*nvar

# Get BW by cross-validation
res = gwrlrp_aicc(y, x, east, north)

# Fitting the model using bw = 20
pars.bwidth = 20
bw_cv = gwrlrp_aicc(y, x, east, north, pars)



rosas.YIELD
getfield(rosas, "Y")
size(y)
