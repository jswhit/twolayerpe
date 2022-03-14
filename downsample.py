from netCDF4 import Dataset
import numpy as np
nskip = 2
filein = 'twolayerpe_N128_6hrly.nc'
fileout = 'twolayerpe_N128_6hrly_nskip2.nc'
ncin = Dataset(filein)
nc = Dataset(fileout,'w')

nc.theta1 = ncin.theta1
nc.theta2 = ncin.theta2
nc.delth = ncin.delth
nc.grav = ncin.grav
nc.umax = ncin.umax
nc.jetexp = ncin.jetexp
nc.ztop = ncin.ztop
nc.zmid = ncin.zmid
nc.f = ncin.f
nc.L = ncin.L
nc.Nt = ncin.Nt//2
nc.N = ncin.N//2
nc.tdiab = ncin.tdiab
nc.tdrag = ncin.tdrag
nc.dt = ncin.dt
nc.diff_efold = ncin.diff_efold
nc.diff_order = ncin.diff_order
x = nc.createDimension('x',nc.Nt)
y = nc.createDimension('y',nc.Nt)
z = nc.createDimension('z',2)
t = nc.createDimension('t',None)
uvar =\
nc.createVariable('u','float32',('t','z','y','x'),zlib=True)
uvar.units = 'm/s'
vvar =\
nc.createVariable('v','float32',('t','z','y','x'),zlib=True)
vvar.units = 'm/s'
dzvar =\
nc.createVariable('dz','float32',('t','z','y','x'),zlib=True)
dzvar.units = 'm'
xvar = nc.createVariable('x','float32',('x',))
xvar.units = 'meters'
yvar = nc.createVariable('y','float32',('y',))
yvar.units = 'meters'
zvar = nc.createVariable('z','float32',('z',))
zvar.units = 'meters'
tvar = nc.createVariable('t','float32',('t',))
tvar.units = 'seconds'
xvar[:] = ncin['x'][::nskip]
yvar[:] = ncin['y'][::nskip]
zvar[0] = ncin.theta1; zvar[1] = ncin.theta2
ntimes = ncin['t'].shape[0]
print(ntimes,'times')
for nout in range(ntimes):
    print(nout)
    uvar[nout,:,:,:] = ncin['u'][nout,:,::nskip,::nskip]
    vvar[nout,:,:,:] = ncin['v'][nout,:,::nskip,::nskip]
    dzvar[nout,:,:,:] = ncin['dz'][nout,:,::nskip,::nskip]
    tvar[nout] = ncin['t'][nout]
ncin.close(); nc.close()


