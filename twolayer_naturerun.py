from pyfft import Fouriert
from twolayer import TwoLayer
import numpy as np
import os, time

L = 20000.e3

# grid, time step info
N = 64 
dt = 600 # time step in seconds
#N = 128
#dt = 300 # time step in seconds

# get OMP_NUM_THREADS (threads to use) from environment.
threads = int(os.getenv('OMP_NUM_THREADS','1'))
precision = 'float32'

ft = Fouriert(N,L,threads=threads,precision=precision)

# create model instance.
# asymmetric jet (drag on lower layer only)
#model=TwoLayer(ft,dt,umax=8,tdrag1=5*86400,tdrag2=1.e30,tdiab=15*86400)
# symmetric jet (default, eddy stats same in both layers)
model=TwoLayer(ft,dt,umax=8,tdrag1=10*86400,tdrag2=10*86400,tdiab=15*86400)

dtype = model.dtype
hrout = 6
outputinterval = hrout*3600. # output interval 
tmin = 100.*86400. # time to start saving data (in days)
tmax = 500.*86400. # time to stop (in days)
nsteps = int(tmax/outputinterval) # number of time steps to animate
# set number of timesteps to integrate for each call to model.advance
model.timesteps = int(outputinterval/model.dt)

# vort, div initial conditions
dtype = model.dtype
l = np.array(2*np.pi,dtype) / model.ft.L
uref = np.zeros((2,model.ft.Nt,model.ft.Nt),dtype=dtype)
vref = np.zeros((2,model.ft.Nt,model.ft.Nt),dtype=dtype)
if model.tdrag[1] < 1.e10:
    uref[0] = -0.5*model.umax*np.sin(l*model.y)
    uref[1] = 0.5*model.umax*np.sin(l*model.y)
else:
    uref[1] = model.umax*np.sin(l*self.y)
vrtspec, divspec = model.ft.getvrtdivspec(uref, vref)
vrtg = model.ft.spectogrd(vrtspec)
vrtg += np.random.normal(0,2.e-6,size=(2,ft.Nt,ft.Nt)).astype(dtype)
# add isolated blob to upper layer
nexp = 20
x = np.arange(0,2.*np.pi,2.*np.pi/ft.Nt); y = np.arange(0.,2.*np.pi,2.*np.pi/ft.Nt)
x,y = np.meshgrid(x,y)
x = x.astype(dtype); y = y.astype(dtype)
vrtg[1] = vrtg[1]+2.e-6*(np.sin(x/2)**(2*nexp)*np.sin(y)**nexp)
vrtspec = model.ft.grdtospec(vrtg)
divspec = np.zeros(vrtspec.shape, vrtspec.dtype)
dzg = model.nlbalance(vrtspec)
dzspec = model.ft.grdtospec(dzg)
ug,vg = model.ft.getuv(vrtspec,divspec)
vrtspec, divspec = model.ft.getvrtdivspec(ug,vg)
if dzg.min() < 0:
    raise ValueError('negative layer thickness! adjust jet parameters')

savedata = 'twolayerpe_N%s_%shrly_symjet.nc' % (N,hrout) # save data plotted in a netcdf file.
#savedata = None # don't save data

if savedata is not None:
    from netCDF4 import Dataset
    nc = Dataset(savedata, mode='w', format='NETCDF4_CLASSIC')
    nc.theta1 = model.theta1
    nc.theta2 = model.theta2
    nc.delth = model.delth
    nc.grav = model.grav
    nc.umax = model.umax
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
    nc.createVariable('u',dtype,('t','z','y','x'),zlib=True)
    uvar.units = 'm/s'
    vvar =\
    nc.createVariable('v',dtype,('t','z','y','x'),zlib=True)
    vvar.units = 'm/s'
    dzvar =\
    nc.createVariable('dz',dtype,('t','z','y','x'),zlib=True)
    dzvar.units = 'm'
    xvar = nc.createVariable('x',dtype,('x',))
    xvar.units = 'meters'
    yvar = nc.createVariable('y',dtype,('y',))
    yvar.units = 'meters'
    zvar = nc.createVariable('z',dtype,('z',))
    zvar.units = 'meters'
    tvar = nc.createVariable('t',dtype,('t',))
    tvar.units = 'seconds'
    xvar[:] = model.x[0,:]
    yvar[:] = model.y[:,0]
    zvar[0] = model.theta1; zvar[1] = model.theta2

t = 0.; nout = 0
t1 = time.perf_counter()
while t < tmax:
    vrtspec, divspec, dzspec = model.advance(vrtspec, divspec, dzspec)
    t = model.t
    th = t/3600.
    print('t = %g hours' % th)
    if savedata is not None and t >= tmin:
        ug, vg = model.ft.getuv(vrtspec, divspec)
        dzg = model.ft.spectogrd(dzspec)
        print(dzg[0].mean(), dzg[1].mean())
        uvar[nout,:,:,:] = ug
        vvar[nout,:,:,:] = vg
        dzvar[nout,:,:,:] = dzg
        tvar[nout] = t
        nc.sync()
        if t >= tmax: nc.close()
        nout = nout + 1
t2 = time.perf_counter()
print('total time = ',t2-t1)
