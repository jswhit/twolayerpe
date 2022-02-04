import numpy as np
from netCDF4 import Dataset
import sys, time, os
from twolayer import TwoLayer
from pyfft import Fouriert
from enkf_utils import cartdist,letkf_update,serial_update,gaspcohn

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
   vertical covariance length scale implied by scaling with Rossby radius.
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
threads = int(os.getenv('OMP_NUM_THREADS','1'))

diff_efold = None # use diffusion from climo file

profile = False # turn on profiling?

use_letkf = True  # if False, use serial EnSRF
read_restart = False
savedata = None # if not None, netcdf filename to save data.
#savedata = True # filename given by exptname env var
nassim = 400 # assimilation times to run

nanals = 20 # ensemble members

oberrstdev = 100. # interface height ob error in meters

# nature run created using twolayer_naturerun.py.
filename_climo = 'twolayerpe_N64_6hrly.nc' # file name for forecast model climo
# perfect model
filename_truth = filename_climo

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
jetexp = nc_climo.jetexp
umax = nc_climo.umax
theta1 = nc_climo.theta1
theta2 = nc_climo.theta2
zmid = nc_climo.zmid
ztop = nc_climo.ztop
tdrag = nc_climo.tdrag
tdiab = nc_climo.tdiab
hmax = nc_climo.hmax
N = nc_climo.N
Nt = nc_climo.Nt
L = nc_climo.L
ft = Fouriert(N,L,threads=threads) # create Fourier transform object
dt = nc_climo.dt
diff_efold=nc_climo.diff_efold
diff_order=nc_climo.diff_order
uens = np.empty((nanals,2,Nt,Nt),np.float32)
vens = np.empty((nanals,2,Nt,Nt),np.float32)
dzens = np.empty((nanals,2,Nt,Nt),np.float32)
if not read_restart:
    u_climo = nc_climo.variables['u']
    v_climo = nc_climo.variables['v']
    dz_climo = nc_climo.variables['dz']
    indxran = rsics.choice(u_climo.shape[0],size=nanals,replace=False)
else:
    ncinit = Dataset('%s_restart.nc' % exptname, mode='r', format='NETCDF4_CLASSIC')
    ncinit.set_auto_mask(False)
    uens[:] = ncinit.variables['v_b'][-1,...]
    vens[:] = ncinit.variables['v_b'][-1,...]
    dzens[:] = ncinit.variables['dz_b'][-1,...]
    tstart = ncinit.variables['t'][-1]
    #for nanal in range(nanals):
    #    print(nanal, uens[nanal].min(), uens[nanal].max())

models = []
for nanal in range(nanals):
    if not read_restart:
        uens[nanal] = u_climo[indxran[nanal]]
        vens[nanal] = v_climo[indxran[nanal]]
        dzens[nanal] = dz_climo[indxran[nanal]]
        #print(nanal, uens[nanal].min(), uens[nanal].max())
    models.append(TwoLayer(ft,dt,zmid=zmid,ztop=ztop,tdrag=tdrag,tdiab=tdiab,\
    hmax=hmax,umax=umax,jetexp=jetexp,theta1=theta1,theta2=theta2,diff_efold=diff_efold))
if read_restart: ncinit.close()

print("# hcovlocal=%g use_letkf=%s covinf1=%s covinf2=%s nanals=%s" %\
     (hcovlocal_scale/1000.,use_letkf,covinflate1,covinflate2,nanals))

# each ob time nobs ob locations are randomly sampled (without
# replacement) from the model grid
#nobs = Nt**2 # observe full grid
nobs = Nt**2//4 # 

# nature run
nc_truth = Dataset(filename_truth)
u_truth = nc_truth.variables['u']
v_truth = nc_truth.variables['v']
dz_truth = nc_truth.variables['dz']

# set up arrays for obs and localization function
print('# random network nobs = %s' % nobs)
oberrvar = oberrstdev**2*np.ones(nobs,np.float32)
zmidobs = np.empty(nobs,np.float32)
covlocal = np.empty(Nt**2,np.float32)
covlocal_tmp = np.empty((nobs,Nt**2),np.float32)
xens = np.empty((nanals,6,Nt**2),np.float32)
if not use_letkf:
    obcovlocal = np.empty((nobs,nobs),np.float32)
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
assim_timesteps = int(np.round(assim_interval/models[0].dt))
print('# ntime,zmiderr,zmidsprd,v2err,v2sprd,zsfcerr,zsfcsprd,v1err,v1sprd,obfits,obsprdplusr')

# initialize model clock
for nanal in range(nanals):
    models[nanal].t = obtimes[ntstart]
    models[nanal].timesteps = assim_timesteps

# initialize output file.
if savedata is not None:
    nc = Dataset('%s.nc' % exptname, mode='w', format='NETCDF4_CLASSIC')
    nc.theta1 = theta1
    nc.theta2 = theta2
    nc.delth = theta2-theta1
    nc.grav = models[0].grav
    nc.umax = umax
    nc.jetexp = jetexp
    nc.hmax = hmax
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
    obs = nc.createDimension('obs',nobs)
    ens = nc.createDimension('ens',nanals)
    u_t =\
    nc.createVariable('u_t',np.float32,('t','z','y','x'),zlib=True)
    u_b =\
    nc.createVariable('u_b',np.float32,('t','ens','z','y','x'),zlib=True)
    u_a =\
    nc.createVariable('u_a',np.float32,('t','ens','z','y','x'),zlib=True)
    v_t =\
    nc.createVariable('v_t',np.float32,('t','z','y','x'),zlib=True)
    v_b =\
    nc.createVariable('v_b',np.float32,('t','ens','z','y','x'),zlib=True)
    v_a =\
    nc.createVariable('v_a',np.float32,('t','ens','z','y','x'),zlib=True)
    dz_t =\
    nc.createVariable('dz_t',np.float32,('t','z','y','x'),zlib=True)
    dz_b =\
    nc.createVariable('dz_b',np.float32,('t','ens','z','y','x'),zlib=True)
    dz_a =\
    nc.createVariable('dz_a',np.float32,('t','ens','z','y','x'),zlib=True)

    obs = nc.createVariable('obs',np.float32,('t','obs'))
    xvar = nc.createVariable('x',np.float32,('x',))
    xvar.units = 'meters'
    yvar = nc.createVariable('y',np.float32,('y',))
    yvar.units = 'meters'
    zvar = nc.createVariable('z',np.float32,('z',))
    zvar.units = 'meters'
    tvar = nc.createVariable('t',np.float32,('t',))
    tvar.units = 'seconds'
    ensvar = nc.createVariable('ens',np.int32,('ens',))
    ensvar.units = 'dimensionless'
    xvar[:] = models[0].x[:]
    yvar[:] = models[0].y[:]
    zvar[0] = models[0].theta1; zvar[1] = models[0].theta2
    ensvar[:] = np.arange(1,nanals+1)

# calculate spread/error stats in model space
def getspreaderr(uens,vens,dzens,u_truth,v_truth,dz_truth,ztop):
    nanals = uens.shape[0]
    uensmean = uens.mean(axis=0)
    uerr = ((uensmean-u_truth))**2
    uprime = uens-uensmean
    usprd = (uprime**2).sum(axis=0)/(nanals-1)
    vensmean = vens.mean(axis=0)
    verr = ((vensmean-v_truth))**2
    vprime = vens-vensmean
    vsprd = (vprime**2).sum(axis=0)/(nanals-1)
    vecwind_err = np.sqrt(uerr+verr)
    vecwind_sprd = np.sqrt(usprd+vsprd)
    zsfc = ztop - dzens.sum(axis=1)
    zmid = ztop - dzens[:,1,:,:]
    zsfcensmean = zsfc.mean(axis=0)
    zsfcprime = zsfc-zsfcensmean
    zsfcsprd = (zsfcprime**2).sum(axis=0)/(nanals-1)
    zsfc_truth = ztop-dz_truth.sum(axis=0)
    zsfcerr =  (zsfcensmean-zsfc_truth)**2
    zmidensmean = zmid.mean(axis=0)
    zmid_truth = ztop-dz_truth[1,:,:]
    zmiderr = (zmidensmean-zmid_truth)**2
    zmidprime = zmid-zmidensmean
    zmidsprd = (zmidprime**2).sum(axis=0)/(nanals-1)
    vecwind1_errav = vecwind_err[0,...].mean()
    vecwind2_errav = vecwind_err[1,...].mean()
    vecwind1_sprdav = vecwind_sprd[0,...].mean()
    vecwind2_sprdav = vecwind_sprd[1,...].mean()
    zsfc_errav = zsfcerr.mean()
    zsfc_sprdav = zsfcsprd.mean()
    zmid_errav = np.sqrt(zmiderr.mean())
    zmid_sprdav = np.sqrt(zmidsprd.mean())
    zsfc_errav = np.sqrt(zsfcerr.mean())
    zsfc_sprdav = np.sqrt(zsfcsprd.mean())
    return vecwind1_errav,vecwind1_sprdav,vecwind2_errav,vecwind2_sprdav,zsfc_errav,zsfc_sprdav,zmid_errav,zmid_sprdav

# forward operator, ob space stats
def gethofx(xens,obs,indxob,nanals,nobs):
    hxens = np.empty((nanals,nobs),np.float32)
    for nanal in range(nanals):
        hxens[nanal,:] = xens[nanal,...].ravel()[indxob] # interface height obs
    hxensmean = hxens.mean(axis=0)
    obsprd = ((hxens-hxensmean)**2).sum(axis=0)/(nanals-1)
    obfits = obs - hxensmean
    # ob space prior stats.
    obfits_av = np.sqrt((obfits**2).mean())
    obsprd_av = np.sqrt(obsprd.mean() + oberrstdev**2)
    return hxens,obfits_av,obsprd_av

for ntime in range(nassim):

    # check model clock
    if models[0].t != obtimes[ntime+ntstart]:
        raise ValueError('model/ob time mismatch %s vs %s' %\
        (models[0].t, obtimes[ntime+ntstart]))

    t1 = time.time()
    # randomly choose points from model grid
    if nobs == Nt**2: # observe every grid point
        indxob = np.arange(Nt**2)
        np.random.shuffle(indxob) # shuffle needed or serial filter blows up (??)
    else: # random selection of grid points
        indxob = rsobs.choice(Nt**2,nobs,replace=False)
    zmidobs = ztop - dz_truth[ntime+ntstart,1,:,:].ravel()[indxob]
    zmidobs += rsobs.normal(scale=oberrstdev,size=nobs) # add ob errors
    xob = x.ravel()[indxob]
    yob = y.ravel()[indxob]

    # compute covariance localization function for each ob
    for nob in range(nobs):
        dist = cartdist(xob[nob],yob[nob],x,y,nc_climo.L,nc_climo.L)
        covlocal = gaspcohn(dist/hcovlocal_scale)
        covlocal_tmp[nob] = covlocal.ravel()
        dist = cartdist(xob[nob],yob[nob],xob,yob,nc_climo.L,nc_climo.L)
        if not use_letkf: obcovlocal[nob] = gaspcohn(dist/hcovlocal_scale)

    # compute forward operator.
    # hxens is ensemble in observation space.
    hxens, obfits_b, obsprd_b = gethofx(ztop-dzens[:,1,...],zmidobs,indxob,nanals,nobs)

    if savedata is not None:
        u_t[ntime] = u_truth[ntime+ntstart]
        u_b[ntime,:,:,:] = uens
        v_t[ntime] = v_truth[ntime+ntstart]
        v_b[ntime,:,:,:] = vens
        dz_t[ntime] = dz_truth[ntime+ntstart]
        dz_b[ntime,:,:,:] = dzens
        obs[ntime] = zmidobs
        x_obs[ntime] = xob
        y_obs[ntime] = yob

    # EnKF update
    # create state vector.
    xens[:,0:2,:] = uens.reshape(nanals,2,Nt**2)
    xens[:,2:4,:] = vens.reshape(nanals,2,Nt**2)
    xens[:,4:6,:] = dzens.reshape(nanals,2,Nt**2)
    xensmean_b = xens.mean(axis=0)
    xprime = xens-xensmean_b
    fsprd = (xprime**2).sum(axis=0)/(nanals-1)

    # prior stats.
    vecwind1_errav_b,vecwind1_sprdav_b,vecwind2_errav_b,vecwind2_sprdav_b,\
    zsfc_errav_b,zsfc_sprdav_b,zmid_errav_b,zmid_sprdav_b=getspreaderr(uens,vens,dzens,\
    u_truth[ntime+ntstart],v_truth[ntime+ntstart],dz_truth[ntime+ntstart],ztop)
    print("%s %g %g %g %g %g %g %g %g %g %g" %\
    (ntime+ntstart,zmid_errav_b,zmid_sprdav_b,vecwind2_errav_b,vecwind2_sprdav_b,\
     zsfc_errav_b,zsfc_sprdav_b,vecwind1_errav_b,vecwind1_sprdav_b,obfits_b,obsprd_b))

    # update state vector with serial filter or letkf.
    if use_letkf:
        xens = letkf_update(xens,hxens,zmidobs,oberrvar,covlocal_tmp)
    else:
        xens = serial_update(xens,hxens,zmidobs,oberrvar,covlocal_tmp,obcovlocal)
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
    uens = xens[:,0:2,:].reshape((nanals,2,Nt,Nt))
    vens = xens[:,2:4,:].reshape((nanals,2,Nt,Nt))
    dzens = xens[:,4:6,:].reshape((nanals,2,Nt,Nt))

    # posterior stats
    #hxens, obfits_a, obsprd_a = gethofx(ztop-dzens[:,1,...],zmidobs,indxob,nanals,nobs)
    #vecwind1_errav_a,vecwind1_sprdav_a,vecwind2_errav_a,vecwind2_sprdav_a,\
    #zsfc_errav_a,zsfc_sprdav_a,zmid_errav_a,zmid_sprdav_a=getspreaderr(uens,vens,dzens,\
    #u_truth[ntime+ntstart],v_truth[ntime+ntstart],dz_truth[ntime+ntstart],ztop)
    #print("%s %g %g %g %g %g %g %g %g %g %g" %\
    #(ntime+ntstart,zmid_errav_a,zmid_sprdav_a,vecwind2_errav_a,vecwind2_sprdav_a,\
    # zsfc_errav_a,zsfc_sprdav_a,vecwind1_errav_a,vecwind1_sprdav_a,obfits_a,obsprd_a))

    # save data.
    if savedata is not None:
        u_a[ntime,:,:,:] = uens
        v_a[ntime,:,:,:] = vens
        dz_a[ntime,:,:,:] = dzens
        tvar[ntime] = obtimes[ntime+ntstart]
        nc.sync()

    # run forecast ensemble to next analysis time
    t1 = time.time()
    for nanal in range(nanals): # TODO: parallelize this embarassingly parallel loop
        vrtspec, divspec = ft.getvrtdivspec(uens[nanal],vens[nanal])
        dzspec = ft.grdtospec(dzens[nanal])
        vrtspec, divspec, dzspec = models[nanal].advance(vrtspec,divspec,dzspec)
        uens[nanal],vens[nanal] = ft.getuv(vrtspec,divspec)
        dzens[nanal] = ft.spectogrd(dzspec)
    t2 = time.time()
    if profile: print('cpu time for ens forecast',t2-t1)

if savedata: nc.close()
