from pyfft import Fouriert
from twolayer import TwoLayer
import numpy as np
import os, time

# grid, time step info
N = 64 
L = 20000.e3
dt = 600 # time step in seconds

# get OMP_NUM_THREADS (threads to use) from environment.
threads = int(os.getenv('OMP_NUM_THREADS','1'))
ft = Fouriert(N,L,threads=threads)

# create model instance, override default parameters.
model=TwoLayer(ft,dt,umax=15,jetexp=8,theta2=315,diff_efold=6.*3600.)

hrout = 6
outputinterval = hrout*3600. # output interval 
tmin = 100.*86400. # time to start saving data (in days)
tmax = 300.*86400. # time to stop (in days)
nsteps = int(tmax/outputinterval) # number of time steps to animate
# set number of timesteps to integrate for each call to model.advance
model.timesteps = int(outputinterval/model.dt)

# vort, div initial conditions
vref = np.zeros(model.uref.shape, model.uref.dtype)
vrtspec, divspec = model.ft.getvrtdivspec(model.uref, vref)
vrtg = model.ft.spectogrd(vrtspec)
vrtg += np.random.normal(0,2.e-6,size=(2,ft.Nt,ft.Nt)).astype(np.float32)
# add isolated blob to upper layer
nexp = 20
x = np.arange(0,2.*np.pi,2.*np.pi/ft.Nt); y = np.arange(0.,2.*np.pi,2.*np.pi/ft.Nt)
x,y = np.meshgrid(x,y)
x = x.astype(np.float32); y = y.astype(np.float32)
vrtg[1] = vrtg[1]+2.e-6*(np.sin(x/2)**(2*nexp)*np.sin(y)**nexp)
vrtspec = model.ft.grdtospec(vrtg)
divspec = np.zeros(vrtspec.shape, vrtspec.dtype)
lyrthkspec = model.nlbalance(vrtspec)
lyrthkg = model.ft.spectogrd(lyrthkspec)
ug,vg = model.ft.getuv(vrtspec,divspec)
vrtspec, divspec = model.ft.getvrtdivspec(ug,vg)
if lyrthkg.min() < 0:
    raise ValueError('negative layer thickness! adjust jet parameters')

savedata = 'twolayerpe_N%s_%shrly.nc' % (N,hrout) # save data plotted in a netcdf file.
#savedata = None # don't save data

if savedata is not None:
    from netCDF4 import Dataset
    nc = Dataset(savedata, mode='w', format='NETCDF4_CLASSIC')
    nc.theta1 = model.theta1
    nc.theta2 = model.theta2
    nc.delth = model.delth
    nc.grav = model.grav
    nc.umax = model.umax
    nc.jetexp = model.jetexp
    nc.hmax = model.hmax
    nc.ztop = model.ztop
    nc.zmid = model.zmid
    nc.f = model.f
    nc.L = model.ft.L
    nc.Nt = model.ft.Nt
    nc.N = model.ft.N
    nc.tdiab = model.tdiab
    nc.tdrag = model.tdrag
    nc.dt = model.dt
    nc.diff_efold = model.diff_efold
    nc.diff_order = model.diff_order
    x = nc.createDimension('x',model.ft.Nt)
    y = nc.createDimension('y',model.ft.Nt)
    z = nc.createDimension('z',2)
    t = nc.createDimension('t',None)
    uvar =\
    nc.createVariable('u',np.float32,('t','z','y','x'),zlib=True)
    uvar.units = 'm/s'
    vvar =\
    nc.createVariable('v',np.float32,('t','z','y','x'),zlib=True)
    vvar.units = 'm/s'
    dzvar =\
    nc.createVariable('dz',np.float32,('t','z','y','x'),zlib=True)
    dzvar.units = 'm'
    xvar = nc.createVariable('x',np.float32,('x',))
    xvar.units = 'meters'
    yvar = nc.createVariable('y',np.float32,('y',))
    yvar.units = 'meters'
    zvar = nc.createVariable('z',np.float32,('z',))
    zvar.units = 'meters'
    tvar = nc.createVariable('t',np.float32,('t',))
    tvar.units = 'seconds'
    xvar[:] = model.x[0,:]
    yvar[:] = model.y[:,0]
    zvar[0] = model.theta1; zvar[1] = model.theta2

t = 0.; nout = 0
t1 = time.perf_counter()
while t < tmax:
    vrtspec, divspec, lyrthkspec = model.advance(vrtspec, divspec, lyrthkspec)
    t = model.t
    th = t/3600.
    print('t = %g hours' % th)
    if savedata is not None and t >= tmin:
        ug, vg = model.ft.getuv(vrtspec, divspec)
        lyrthkg = model.ft.spectogrd(lyrthkspec)
        uvar[nout,:,:,:] = ug
        vvar[nout,:,:,:] = vg
        dzvar[nout,:,:,:] = lyrthkg
        tvar[nout] = t
        nc.sync()
        if t >= tmax: nc.close()
        nout = nout + 1
t2 = time.perf_counter()
print('total time = ',t2-t1)
