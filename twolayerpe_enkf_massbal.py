import numpy as np
from netCDF4 import Dataset
import sys, time, os
from twolayer import TwoLayer, run_model
from pyfft import Fouriert
from enkf_utils import cartdist,letkf_update,serial_update,gaspcohn,letkfwts_compute2
from joblib import Parallel, delayed

# EnKF cycling for two-layer pe turbulence model with interface height obs.
# horizontal localization (no vertical).
# Relaxation to prior spread
# inflation, or Hodyss and Campbell inflation.
# random observing network.
# Options for serial EnSRF or LETKF.

if len(sys.argv) == 1:
   msg="""
python twolayerpe_enkf.py hcovlocal_scale <covinflate1 covinflate2>
   hcovlocal_scale = horizontal localization scale in km
   no vertical localization
   covinflate1,covinflate2: inflation parameters (optional).
   if only covinflate1 is specified, it is interpreted as the relaxation
   factor for RTPS inflation.
   if neither covinflate1 or covinflate2 specified
   Hodyss et al inflation (http://journals.ametsoc.org/doi/abs/10.1175/MWR-D-15-0329.1)
   with a=b=1 used.
   if both covinflate1 and covinflate2 given, they correspond to a and b in the
   Hodyss et al inflation (eqn 4.4).
   """
   raise SystemExit(msg)

# horizontal covariance localization length scale in meters.
hcovlocal_scale = 1.e3*float(sys.argv[1])

# optional inflation parameters:
# (covinflate2 <= 0 for RTPS inflation
# (http://journals.ametsoc.org/doi/10.1175/MWR-D-11-00276.1),
# otherwise use Hodyss et al inflation
# (http://journals.ametsoc.org/doi/abs/10.1175/MWR-D-15-0329.1)
if len(sys.argv) == 3:
    covinflate1 = float(sys.argv[2])
    covinflate2 = -1
elif len(sys.argv) == 4:
    covinflate1 = float(sys.argv[2])
    covinflate2 = float(sys.argv[3])
else:
    covinflate1 = 1.
    covinflate2 = 1.

exptname = os.getenv('exptname','test')
# get envar to set number of multiprocessing jobs for LETKF and ensemble forecast
n_jobs = int(os.getenv('N_JOBS','0'))
threads = 1

profile = False # turn on profiling?

use_letkf = True # if False, use serial EnSRF
fix_totmass = False # if True, use a mass fixer to fix mass in each layer (area mean dz)
read_restart = False
debug_model = False # run perfect model ensemble, check to see that error=zero with no DA
precision = 'float32'
savedata = None # if not None, netcdf filename to save data.
#savedata = True # filename given by exptname env var
nassim = 1600 # assimilation times to run
ntime_savestart = 600 # if savedata is not None, start saving data at this time
dzmin = 10. # min layer thickness allowed

nanals = 20 # ensemble members

# nature run created using twolayer_naturerun.py.
filename_climo = 'twolayerpe_N64_6hrly_symjet.nc' # file name for forecast model climo
# perfect model
#filename_truth = filename_climo
filename_truth = 'twolayerpe_N128_6hrly_symjet_nskip2.nc' # file name for forecast model climo

print('# filename_modelclimo=%s' % filename_climo)
print('# filename_truth=%s' % filename_truth)

# fix random seed for reproducibility.
rsobs = np.random.RandomState(42) # fixed seed for observations
rsics = np.random.RandomState() # varying seed for initial conditions

# get model info
nc_climo = Dataset(filename_climo)

# initialize model instances for each ensemble member.
x = nc_climo.variables['x'][:]
y = nc_climo.variables['y'][:]
x, y = np.meshgrid(x, y)
umax = nc_climo.umax
theta1 = nc_climo.theta1
theta2 = nc_climo.theta2
f = nc_climo.f
zmid = nc_climo.zmid
ztop = nc_climo.ztop
tdrag = nc_climo.tdrag
tdiab = nc_climo.tdiab
N = nc_climo.N
Nt = nc_climo.Nt
L = nc_climo.L
dt = nc_climo.dt
diff_efold=nc_climo.diff_efold
diff_order=nc_climo.diff_order

oberrstdev_zmid = 100. # interface height ob error in meters
oberrstdev_zsfc = ((theta2-theta1)/theta1)*100.  # surface height ob error in meters
#oberrstdev_wind = 1.   # wind ob error in meters per second
#oberrstdev_zsfc = 1.e30 # surface height ob error in meters
oberrstdev_wind = 1.e30 # don't assimilate winds

ft = Fouriert(N,L,threads=threads,precision=precision) # create Fourier transform object

#div2_diff_efold=1800.
div2_diff_efold=1.e30
model = TwoLayer(ft,dt,zmid=zmid,ztop=ztop,tdrag1=tdrag[0],tdrag2=tdrag[1],tdiab=tdiab,\
umax=umax,theta1=theta1,theta2=theta2,diff_efold=diff_efold,\
div2_diff_efold=div2_diff_efold)
if debug_model:
   print('N,Nt,L=',N,Nt,L)
   print('theta1,theta2=',theta1,theta2)
   print('zmid,ztop=',zmid,ztop)
   print('tdrag,tdiag=',tdrag/86400,tdiab/86400.)
   print('umax=',umax)
   print('diff_order,diff_efold=',diff_order,diff_efold)

dtype = model.dtype
uens = np.empty((nanals,2,Nt,Nt),dtype)
vens = np.empty((nanals,2,Nt,Nt),dtype)
dzens = np.empty((nanals,2,Nt,Nt),dtype)
if not read_restart:
    u_climo = nc_climo.variables['u']
    v_climo = nc_climo.variables['v']
    dz_climo = nc_climo.variables['dz']
    indxran = rsics.choice(u_climo.shape[0],size=nanals,replace=False)
else:
    ncinit = Dataset('%s.nc' % exptname, mode='r', format='NETCDF4_CLASSIC')
    ncinit.set_auto_mask(False)
    uens[:] = ncinit.variables['u_b'][-1,...]
    vens[:] = ncinit.variables['v_b'][-1,...]
    dzens[:] = ncinit.variables['dz_b'][-1,...]
    tstart = ncinit.variables['t'][-1]
    #for nanal in range(nanals):
    #    print(nanal, uens[nanal].min(), uens[nanal].max())

if not read_restart:
    if debug_model:
        for nanal in range(nanals):
            uens[nanal] = u_climo[0]
            vens[nanal] = v_climo[0]
            dzens[nanal] = dz_climo[0]
    else:
        for nanal in range(nanals):
            uens[nanal] = u_climo[indxran[nanal]]
            vens[nanal] = v_climo[indxran[nanal]]
            dzens[nanal] = dz_climo[indxran[nanal]]
            #print(nanal, uens[nanal].min(), uens[nanal].max())
else:
    ncinit.close()

print("# hcovlocal=%g use_letkf=%s covinf1=%s covinf2=%s nanals=%s" %\
     (hcovlocal_scale/1000.,use_letkf,covinflate1,covinflate2,nanals))

# each ob time nobs ob locations are randomly sampled (without
# replacement) from the model grid
#nobs = Nt**2 # observe full grid
nobs = Nt**2//16

# nature run
nc_truth = Dataset(filename_truth)
u_truth = nc_truth.variables['u']
v_truth = nc_truth.variables['v']
dz_truth = nc_truth.variables['dz']

# set up arrays for obs and localization function
print('# random network nobs = %s' % nobs)
oberrvar = np.ones(6*nobs,dtype)
oberrvar[0:4*nobs] = oberrstdev_wind*oberrvar[0:4*nobs]
oberrvar[4*nobs:5*nobs] = oberrstdev_zsfc*oberrvar[4*nobs:5*nobs]
oberrvar[5*nobs:] = oberrstdev_zmid*oberrvar[5*nobs:]

obs = np.empty(6*nobs,dtype)
covlocal1 = np.empty(Nt**2,dtype)
covlocal1_tmp = np.empty((nobs,Nt**2),dtype)
covlocal_tmp = np.empty((6*nobs,Nt**2),dtype)
if not use_letkf:
    obcovlocal1 = np.empty((nobs,nobs),dtype)
else:
    obcovlocal = None

obtimes = nc_truth.variables['t'][:]
if read_restart:
    timeslist = obtimes.tolist()
    ntstart = timeslist.index(tstart)
    print('# restarting from %s.nc ntstart = %s' % (exptname,ntstart))
else:
    ntstart = 0
assim_interval = obtimes[1]-obtimes[0]
assim_timesteps = int(np.round(assim_interval/model.dt))
print('# div2_diff_efold = %s' % div2_diff_efold)
print('# oberrzsfc=%s oberrzmid=%s oberrwind=%s' % (oberrstdev_zsfc,oberrstdev_zmid,oberrstdev_wind))
print('# assim_interval = %s assim_timesteps = %s' % (assim_interval,assim_timesteps))

# initialize model clock
model.t = obtimes[ntstart]
model.timesteps = assim_timesteps

# initialize output file.
if savedata is not None:
    nc = Dataset('%s.nc' % exptname, mode='w', format='NETCDF4_CLASSIC')
    nc.dt = dt
    nc.precision = precision
    nc.theta1 = theta1
    nc.theta2 = theta2
    nc.delth = theta2-theta1
    nc.grav = model.grav
    nc.umax = umax
    nc.ztop = ztop
    nc.zmid = zmid
    nc.f = model.f
    nc.L = ft.L
    nc.Nt =ft.Nt
    nc.N = ft.N
    nc.tdiab = tdiab
    nc.tdrag = tdrag
    nc.diff_efold = diff_efold
    nc.diff_order = diff_order
    nc.filename_climo = filename_climo
    nc.filename_truth = filename_truth
    xdim = nc.createDimension('x',Nt)
    ydim = nc.createDimension('y',Nt)
    z = nc.createDimension('z',2)
    t = nc.createDimension('t',None)
    obsd = nc.createDimension('obs',nobs)
    ens = nc.createDimension('ens',nanals)
    u_t =\
    nc.createVariable('u_t',dtype,('t','z','y','x'),zlib=True)
    u_b =\
    nc.createVariable('u_b',dtype,('t','ens','z','y','x'),zlib=True)
    u_a =\
    nc.createVariable('u_a',dtype,('t','ens','z','y','x'),zlib=True)
    v_t =\
    nc.createVariable('v_t',dtype,('t','z','y','x'),zlib=True)
    v_b =\
    nc.createVariable('v_b',dtype,('t','ens','z','y','x'),zlib=True)
    v_a =\
    nc.createVariable('v_a',dtype,('t','ens','z','y','x'),zlib=True)
    dz_t =\
    nc.createVariable('dz_t',dtype,('t','z','y','x'),zlib=True)
    dz_b =\
    nc.createVariable('dz_b',dtype,('t','ens','z','y','x'),zlib=True)
    dz_a =\
    nc.createVariable('dz_a',dtype,('t','ens','z','y','x'),zlib=True)

    obsu1 = nc.createVariable('obsu1',dtype,('t','obs'))
    obsv1 = nc.createVariable('obsv1',dtype,('t','obs'))
    obsu2 = nc.createVariable('obsu2',dtype,('t','obs'))
    obsv2 = nc.createVariable('obsv2',dtype,('t','obs'))
    obszsfc = nc.createVariable('obszsfc',dtype,('t','obs'))
    obszmid = nc.createVariable('obszmid',dtype,('t','obs'))
    x_obs = nc.createVariable('x_ob',dtype,('t','obs'))
    y_obs = nc.createVariable('y_ob',dtype,('t','obs'))
    xvar = nc.createVariable('x',dtype,('x',))
    xvar.units = 'meters'
    yvar = nc.createVariable('y',dtype,('y',))
    yvar.units = 'meters'
    zvar = nc.createVariable('z',dtype,('z',))
    zvar.units = 'meters'
    tvar = nc.createVariable('t',dtype,('t',))
    tvar.units = 'seconds'
    ensvar = nc.createVariable('ens',np.int32,('ens',))
    ensvar.units = 'dimensionless'
    xvar[:] = model.x[0,:]
    yvar[:] = model.y[:,0]
    zvar[0] = model.theta1; zvar[1] = model.theta2
    ensvar[:] = np.arange(1,nanals+1)

# calculate spread/error stats in model space
def getspreaderr(model,uens,vens,dzens,u_truth,v_truth,dz_truth,ztop):
    nanals = uens.shape[0]
    uensmean = uens.mean(axis=0)
    uerr = ((uensmean-u_truth))**2
    uprime = uens-uensmean
    usprd = (uprime**2).sum(axis=0)/(nanals-1)
    vensmean = vens.mean(axis=0)
    verr = ((vensmean-v_truth))**2
    vprime = vens-vensmean
    vsprd = (vprime**2).sum(axis=0)/(nanals-1)
    ke_err = uerr+verr
    ke_sprd = usprd+vsprd
    vecwind_err = np.sqrt(uerr+verr)
    vecwind_sprd = np.sqrt(usprd+vsprd)

    zsfc = model.ztop - dzens.sum(axis=1)
    zmid = model.ztop - dzens[:,1,:,:]
    zsfc_truth = model.ztop-dz_truth.sum(axis=0)
    zmid_truth = model.ztop-dz_truth[1,:,:]

    # montgomery potential (divided by g, units of m)
    m1 = dzens.sum(axis=1)
    m2 = m1 + (model.delth/model.theta1)*dzens[:,1,...]
    m1_truth = dz_truth[0,...]+dz_truth[1,...]
    m2_truth = m1_truth + (model.delth/model.theta1)*dz_truth[1,...]

    zsfcensmean = zsfc.mean(axis=0)
    zmidensmean = zmid.mean(axis=0)
    m1ensmean = m1.mean(axis=0)
    m2ensmean = m2.mean(axis=0)
    zmiderr = (zmidensmean-zmid_truth)**2
    zmidprime = zmid-zmidensmean
    zmidsprd = (zmidprime**2).sum(axis=0)/(nanals-1)
    m1err = (m1ensmean-m1_truth)**2
    m1prime = m1-m1ensmean
    m1sprd = (m1prime**2).sum(axis=0)/(nanals-1)
    m2err = (m2ensmean-m2_truth)**2
    m2prime = m2-m2ensmean
    m2sprd = (m2prime**2).sum(axis=0)/(nanals-1)
    zsfcerr =  (zsfcensmean-zsfc_truth)**2
    zsfcprime = zsfc-zsfcensmean
    zsfcsprd = (zsfcprime**2).sum(axis=0)/(nanals-1)

    vecwind1_errav = vecwind_err[0,...].mean()
    vecwind2_errav = vecwind_err[1,...].mean()
    vecwind1_sprdav = vecwind_sprd[0,...].mean()
    vecwind2_sprdav = vecwind_sprd[1,...].mean()
    ke_errav = np.sqrt(0.5*(ke_err[0].mean()+ke_err[1].mean()))
    ke_sprdav = np.sqrt(0.5*(ke_sprd[0].mean()+ke_sprd[1].mean()))
    zmid_errav = np.sqrt(zmiderr.mean())
    zmid_sprdav = np.sqrt(zmidsprd.mean())
    m1_errav = np.sqrt(m1err.mean())
    m1_sprdav = np.sqrt(m1sprd.mean())
    m2_errav = np.sqrt(m2err.mean())
    m2_sprdav = np.sqrt(m2sprd.mean())
    m_errav = np.sqrt(0.5*(m1err.mean() + m2err.mean()))
    m_sprdav = np.sqrt(0.5*(m1sprd.mean() + m2sprd.mean()))
    zsfc_errav = np.sqrt(zsfcerr.mean())
    zsfc_sprdav = np.sqrt(zsfcsprd.mean())

    #return zsfc_errav,zsfc_sprdav,zmid_errav,zmid_sprdav,m1_errav,m1_sprdav,m2_errav,m2_sprdav,vecwind1_errav,vecwind1_sprdav,vecwind2_errav,vecwind2_sprdav
    return zsfc_errav,zsfc_sprdav,zmid_errav,zmid_sprdav,m_errav,m_sprdav,ke_errav,ke_sprdav

# forward operator, ob space stats
def gethofx(uens,vens,zsfcens,zmidens,indxob,nanals,nobs):
    hxens = np.empty((nanals,6*nobs),dtype)
    for nanal in range(nanals):
        hxens[nanal,0:nobs] = uens[nanal,0,...].ravel()[indxob] # interface height obs
        hxens[nanal,nobs:2*nobs] = vens[nanal,0,...].ravel()[indxob] # interface height obs
        hxens[nanal,2*nobs:3*nobs] = uens[nanal,1,...].ravel()[indxob] # interface height obs
        hxens[nanal,3*nobs:4*nobs] = vens[nanal,1,...].ravel()[indxob] # interface height obs
        hxens[nanal,4*nobs:5*nobs] = zsfcens[nanal,...].ravel()[indxob] # interface height obs
        hxens[nanal,5*nobs:] = zmidens[nanal,...].ravel()[indxob] # interface height obs
    return hxens

def enstoctl(model,uens,vens,dzens):
    nanals = uens.shape[0]; Nt = model.ft.Nt
    xens = np.empty((nanals,8,Nt**2),dtype)
    xens[:,0:2,:] = uens[:].reshape(nanals,2,Nt**2)
    xens[:,2:4,:] = vens[:].reshape(nanals,2,Nt**2)
    xens[:,4:6,:] = dzens[:].reshape(nanals,2,Nt**2)
    for nanal in range(nanals):
        umassflux = (uens[nanal]*dzens[nanal]).sum(axis=0)
        vmassflux = (vens[nanal]*dzens[nanal]).sum(axis=0)
        massfluxvrtspec, massfluxdivspec = model.ft.getvrtdivspec(umassflux,vmassflux)
        #massfluxvrtspec = model.ft.invlap*massfluxvrtspec
        #massfluxdivspec = model.ft.invlap*massfluxdivspec
        massfluxdiv = model.ft.spectogrd(massfluxdivspec)
        massfluxvrt = model.ft.spectogrd(massfluxvrtspec)
        xens[nanal,6,:] = massfluxdiv.reshape(model.ft.Nt**2)
        xens[nanal,7,:] = massfluxvrt.reshape(model.ft.Nt**2)
    return xens

def ctltoens(model,xens, xens_b, fsprd):
    # for massbal adjustment see https://doi.org/10.5194/gmd-2020-299
    nanals = xens.shape[0]; Nt = model.ft.Nt
    uens = np.empty((nanals,2,Nt,Nt),dtype)
    vens = np.empty((nanals,2,Nt,Nt),dtype)
    dzens = np.empty((nanals,2,Nt,Nt),dtype)
    uens_b = np.empty((nanals,2,Nt,Nt),dtype)
    vens_b = np.empty((nanals,2,Nt,Nt),dtype)
    uens_sprd = np.empty((2,Nt,Nt),dtype)
    vens_sprd = np.empty((2,Nt,Nt),dtype)
    dzens_b = np.empty((nanals,2,Nt,Nt),dtype)
    uens[:] = xens[:,0:2,:].reshape(nanals,2,Nt,Nt)
    vens[:] = xens[:,2:4,:].reshape(nanals,2,Nt,Nt)
    dzens[:] = xens[:,4:6,:].reshape(nanals,2,Nt,Nt)
    uensmean = uens.mean(axis=0); vensmean = vens.mean(axis=0)
    dzensmean = dzens.mean(axis=0)
    uens_b[:]  = xens_b[:,0:2,:].reshape(nanals,2,Nt,Nt)
    vens_b[:]  = xens_b[:,2:4,:].reshape(nanals,2,Nt,Nt)
    uens_sprd[:]  = fsprd[0:2,:].reshape(2,Nt,Nt)
    vens_sprd[:]  = fsprd[2:4,:].reshape(2,Nt,Nt)
    dzens_sprd[:]  = fsprd[4:6,:].reshape(2,Nt,Nt)
    dzens_b[:] = xens_b[:,4:6,:].reshape(nanals,2,Nt,Nt)
    uensmean_b = uens_b.mean(axis=0); vensmean_b = vens_b.mean(axis=0)
    dzensmean_b = dzens_b.mean(axis=0)
    incmask = np.sqrt(uens_sprd + vens_sprd)
    #incmask = np.sqrt(dzens_sprd)
    #incmask = np.ones((2,Nt,Nt),uens.dtype)
    for nanal in range(nanals):
        umassflux = (uens[nanal]*dzens[nanal]).sum(axis=0)
        vmassflux = (vens[nanal]*dzens[nanal]).sum(axis=0)
        #umassflux_b = (uens_b[nanal]*dzens_b[nanal]).sum(axis=0)
        #vmassflux_b = (vens_b[nanal]*dzens_b[nanal]).sum(axis=0)
        #massfluxvrtspec, massfluxdivspec = model.ft.getvrtdivspec(umassflux,vmassflux)
        #massfluxvrtspec[:]=0
        #massfluxdiv = model.ft.spectogrd(massfluxdivspec)
        #massfluxdiv_new = massfluxdiv.sum(axis=0)
        #print(massfluxdiv_new.min(), massfluxdiv_new.max())
        massfluxdiv_a = xens[nanal,6,:].reshape(Nt,Nt)
        massfluxvrt_a = xens[nanal,7,:].reshape(Nt,Nt)
        massfluxvrtspec = model.ft.grdtospec(massfluxvrt_a)
        massfluxdivspec = model.ft.grdtospec(massfluxdiv_a)
        #massfluxvrtspec = model.ft.lap*massfluxvrtspec
        #massfluxdivspec = model.ft.lap*massfluxdivspec
        umassflux_a, vmassflux_a = model.ft.getuv(massfluxvrtspec, massfluxdivspec)
        # uniform distribution
        #incmask = np.ones((2,Nt,Nt),uens.dtype)
        # proportional to wind increment magnitude
        #incmask = 0.01+np.sqrt((uens[nanal]-uens_b[nanal])**2+(vens[nanal]-vens_b[nanal])**2)
        #incmask = np.abs(dzens[nanal]-dzens_b[nanal])
        #print(incmask.min(), incmask.max())
        uinc = (umassflux - umassflux_a)/(dzens[nanal]*incmask).sum(axis=0)
        uinc = uinc[np.newaxis,:,:]*incmask
        #print(uinc.min(), uinc.max())
        uens[nanal] -= uinc
        vinc = (vmassflux - vmassflux_a)/(dzens[nanal]*incmask).sum(axis=0)
        vinc = vinc[np.newaxis,:,:]*incmask
        #print(vinc.min(), vinc.max())
        vens[nanal] -= vinc
        # recompute
        #umassflux = (uens[nanal]*dzens[nanal]).sum(axis=0)
        #vmassflux = (vens[nanal]*dzens[nanal]).sum(axis=0)
        #massfluxvrtspec, massfluxdivspec = model.ft.getvrtdivspec(umassflux,vmassflux)
        #massfluxdiv_new = model.ft.spectogrd(massfluxdivspec)
        #print(massfluxdiv_new.min(), massfluxdiv_new.max())
        #print(massfluxdiv_a.min(), massfluxdiv_a.max())
        #import matplotlib.pyplot as plt
        #fig = plt.figure()
        #plt.subplot(1,2,1)
        #plt.imshow(massfluxdiv_new,cmap=plt.cm.bwr,interpolation="nearest")
        #plt.title('implied massfluxdiv')
        #plt.colorbar()
        #plt.subplot(1,2,2)
        #plt.imshow(massfluxdiv_a,cmap=plt.cm.bwr,interpolation="nearest")
        #plt.title('analyzed massfluxdiv')
        #plt.colorbar()
        #plt.savefig('test.png')
        #raise SystemExit
        
    return uens,vens,dzens

masstend_diag = 0.
inflation_factor = np.ones((2,Nt,Nt))
for ntime in range(nassim):

    # check model clock
    if model.t != obtimes[ntime+ntstart]:
        raise ValueError('model/ob time mismatch %s vs %s' %\
        (model.t, obtimes[ntime+ntstart]))

    t1 = time.time()
    # randomly choose points from model grid
    if nobs == Nt**2: # observe every grid point
        indxob = np.arange(Nt**2)
        np.random.shuffle(indxob) # shuffle needed or serial filter blows up (??)
    else: # random selection of grid points
        indxob = rsobs.choice(Nt**2,nobs,replace=False)
    obs[0:nobs:] = u_truth[ntime+ntstart,0,:,:].ravel()[indxob] + rsobs.normal(scale=oberrstdev_wind,size=nobs)
    obs[nobs:2*nobs] = v_truth[ntime+ntstart,0,:,:].ravel()[indxob] + rsobs.normal(scale=oberrstdev_wind,size=nobs)
    obs[2*nobs:3*nobs] = u_truth[ntime+ntstart,1,:,:].ravel()[indxob] + rsobs.normal(scale=oberrstdev_wind,size=nobs)
    obs[3*nobs:4*nobs] = v_truth[ntime+ntstart,1,:,:].ravel()[indxob] + rsobs.normal(scale=oberrstdev_wind,size=nobs)
    obs[4*nobs:5*nobs] = ztop - dz_truth[ntime+ntstart,...].sum(axis=0).ravel()[indxob] +\
                   rsobs.normal(scale=oberrstdev_zsfc,size=nobs) 
    obs[5*nobs:] = ztop - dz_truth[ntime+ntstart,1,:,:].ravel()[indxob] +\
                   rsobs.normal(scale=oberrstdev_zmid,size=nobs) 
    xob = x.ravel()[indxob]
    yob = y.ravel()[indxob]

    # compute covariance localization function for each ob
    for nob in range(nobs):
        dist = cartdist(xob[nob],yob[nob],x,y,nc_climo.L,nc_climo.L)
        covlocal1 = gaspcohn(dist/hcovlocal_scale)
        covlocal1_tmp[nob] = covlocal1.ravel()
        dist = cartdist(xob[nob],yob[nob],xob,yob,nc_climo.L,nc_climo.L)
        if not use_letkf: obcovlocal1[nob] = gaspcohn(dist/hcovlocal_scale)
    covlocal_tmp[nobs:2*nobs,:] = covlocal1_tmp
    covlocal_tmp[2*nobs:3*nobs,:] = covlocal1_tmp
    covlocal_tmp[3*nobs:4*nobs,:] = covlocal1_tmp
    covlocal_tmp[4*nobs:5*nobs,:] = covlocal1_tmp
    covlocal_tmp[5*nobs:,:] = covlocal1_tmp
    if not use_letkf:
        obcovlocal = np.block(
            [
                [obcovlocal1,obcovlocal1,obcovlocal1,obcovlocal1,obcovlocal1,obcovlocal1],
                [obcovlocal1,obcovlocal1,obcovlocal1,obcovlocal1,obcovlocal1,obcovlocal1],
                [obcovlocal1,obcovlocal1,obcovlocal1,obcovlocal1,obcovlocal1,obcovlocal1],
                [obcovlocal1,obcovlocal1,obcovlocal1,obcovlocal1,obcovlocal1,obcovlocal1],
                [obcovlocal1,obcovlocal1,obcovlocal1,obcovlocal1,obcovlocal1,obcovlocal1],
                [obcovlocal1,obcovlocal1,obcovlocal1,obcovlocal1,obcovlocal1,obcovlocal1]
            ]
        )

    # compute forward operator.
    # hxens is ensemble in observation space.
    hxens = gethofx(uens,vens,ztop-dzens.sum(axis=1),ztop-dzens[:,1,...],indxob,nanals,nobs)

    if savedata is not None and ntime >= ntime_savestart:
        u_t[ntime-ntime_savestart] = u_truth[ntime+ntstart]
        u_b[ntime-ntime_savestart,:,:,:] = uens
        v_t[ntime-ntime_savestart] = v_truth[ntime+ntstart]
        v_b[ntime-ntime_savestart,:,:,:] = vens
        dz_t[ntime-ntime_savestart] = dz_truth[ntime+ntstart]
        dz_b[ntime-ntime_savestart,:,:,:] = dzens
        obsu1[ntime-ntime_savestart] = obs[0:nobs]
        obsv1[ntime-ntime_savestart] = obs[nobs:2*nobs]
        obsu2[ntime-ntime_savestart] = obs[2*nobs:3*nobs]
        obsu2[ntime-ntime_savestart] = obs[3*nobs:4*nobs]
        obszsfc[ntime-ntime_savestart] = obs[4*nobs:5*nobs]
        obszmid[ntime-ntime_savestart] = obs[5*nobs:]
        x_obs[ntime-ntime_savestart] = xob
        y_obs[ntime-ntime_savestart] = yob

    # EnKF update
    # create state vector.
    xens = enstoctl(model,uens,vens,dzens)
    xens_b = xens.copy()
    xensmean_b = xens.mean(axis=0)
    xprime = xens-xensmean_b
    fsprd = (xprime**2).sum(axis=0)/(nanals-1)

    # prior stats.
    #zsfc_errav,zsfc_sprdav,zmid_errav,zmid_sprdav,m1_errav,m1_sprdav,m2_errav,m2_sprdav,vecwind1_errav,vecwind1_sprdav,vecwind2_errav,vecwind2_sprdav=\
    zsfc_errav,zsfc_sprdav,zmid_errav,zmid_sprdav,m_errav,m_sprdav,ke_errav,ke_sprdav=\
    getspreaderr(model,uens,vens,dzens,\
    u_truth[ntime+ntstart],v_truth[ntime+ntstart],dz_truth[ntime+ntstart],ztop)
    totmass = ((dzens[:,0,...]+dzens[:,1,...]).mean(axis=0)).mean()/1000.
    #print("%s %g %g %g %g %g %g %g %g %g %g %g %g %g %g" %\
    #(ntime+ntstart,zsfc_errav,zsfc_sprdav,zmid_errav,zmid_sprdav,m1_errav,m1_sprdav,m2_errav,m2_sprdav,\
    # vecwind1_errav,vecwind1_spdav,vecwind2_errav,vecwind2_sprdav,masstend_diag,totmass))
    print("%s %g %g %g %g %g %g %g %g %g %g" %\
    (ntime+ntstart,zsfc_errav,zsfc_sprdav,zmid_errav,zmid_sprdav,m_errav,m_sprdav,\
     ke_errav,ke_sprdav,masstend_diag,totmass))

    # update state vector with serial filter or letkf.
    if not debug_model: 
        if use_letkf:
            #xens = letkf_update(xens,hxens,obs,oberrvar,covlocal_tmp,n_jobs)
            wts,wtsmean = letkfwts_compute2(hxens,obs,oberrvar,covlocal_tmp,n_jobs)
            # update state vector using letkf weights
            xprime_a = xprime.copy()
            xensmean_a = xensmean_b.copy()
            for k in range(xens.shape[1]): # only update balanced u,v
                for n in range(model.ft.Nt**2):
                    xensmean_a[k, n] = xensmean_b[k,n] + (wtsmean[n]*xprime[:, k, n]).sum()
                    xprime_a[:, k, n] = np.dot(wts[n].T, xprime[:, k, n])
            xens = xprime_a + xensmean_a
        else:
            xens = serial_update(xens,hxens,obs,oberrvar,covlocal_tmp,obcovlocal)
        t2 = time.time()
        if profile: print('cpu time for EnKF update',t2-t1)

        # posterior multiplicative inflation.
        xensmean_a = xens.mean(axis=0)
        xprime = xens-xensmean_a
        asprd = (xprime**2).sum(axis=0)/(nanals-1)
        if covinflate2 < 0:
            # relaxation to prior stdev (Whitaker & Hamill 2012)
            asprd = np.sqrt(asprd); fsprd = np.sqrt(fsprd)
            inflation_factor = 1.+covinflate1*(fsprd-asprd)/asprd
        else:
            # Hodyss et al 2016 inflation (covinflate1=covinflate2=1 works well in perfect
            # model, linear gaussian scenario)
            # inflation = asprd + (asprd/fsprd)**2((fsprd/nanals)+2*inc**2/(nanals-1))
            inc = xensmean_a - xensmean_b
            inflation_factor = covinflate1*asprd + \
            (asprd/fsprd)**2*((fsprd/nanals) + covinflate2*(2.*inc**2/(nanals-1)))
            inflation_factor = np.sqrt(inflation_factor/asprd)
        xprime = xprime*inflation_factor
        xens = xprime + xensmean_a

    # back to 3d state vector
    uens,vens,dzens = ctltoens(model,xens,xens_b,fsprd)
    np.clip(dzens,a_min=dzmin,a_max=model.ztop-dzmin, out=dzens)
    if fix_totmass:
        for nmem in range(nanals):
            dzens[nmem][0] = dzens[nmem][0] - dzens[nmem][0].mean() + model.zmid
            dzens[nmem][1] = dzens[nmem][1] - dzens[nmem][1].mean() + model.ztop - model.zmid

    # save data.
    if savedata is not None and ntime >= ntime_savestart:
        u_a[ntime-ntime_savestart,:,:,:] = uens
        v_a[ntime-ntime_savestart,:,:,:] = vens
        dz_a[ntime-ntime_savestart,:,:,:] = dzens
        tvar[ntime-ntime_savestart] = obtimes[ntime+ntstart]
        nc.sync()

    # run forecast ensemble to next analysis time
    t1 = time.time()
    tstart = model.t
    if n_jobs == 0:
        masstend_diag=0.
        for nanal in range(nanals): 
            uens[nanal],vens[nanal],dzens[nanal] = model.advance(uens[nanal],vens[nanal],dzens[nanal],grid=True)
            masstend_diag+=model.masstendvar/nanals
    else:
        # use joblib to run ens members on different cores (N_JOBS env var sets number of tasks).
        results = Parallel(n_jobs=n_jobs)(delayed(run_model)(uens[nanal],vens[nanal],dzens[nanal],N,L,dt,assim_timesteps,theta1=theta1,theta2=theta2,zmid=zmid,ztop=ztop,diff_efold=diff_efold,diff_order=diff_order,tdrag1=tdrag[0],tdrag2=tdrag[1],tdiab=tdiab,umax=umax,div2_diff_efold=div2_diff_efold) for nanal in range(nanals))
        masstend_diag=0.
        for nanal in range(nanals):
            uens[nanal],vens[nanal],dzens[nanal],mtend = results[nanal]
            masstend_diag+=mtend/nanals
    model.t = tstart + dt*assim_timesteps
    t2 = time.time()
    if profile: print('cpu time for ens forecast',t2-t1)

if savedata: nc.close()
