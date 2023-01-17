import numpy as np
from netCDF4 import Dataset
import sys, time, os
from twolayer import TwoLayer, run_model
from pyfft import Fouriert
from enkf_utils import cartdist,gaspcohn,getkf,modens
from scipy.linalg import eigh
from joblib import Parallel, delayed

# GETKF cycling for two-layer pe turbulence model.
# model-space horizontal localization (no vertical).
# Relaxation to prior spread inflatino.

if len(sys.argv) == 1:
   msg="""
python twolayerpe_letkf.py hcovlocal_scale covinflate
   hcovlocal_scale = horizontal localization scale in km
   no vertical localization.
   covinflate is the relaxation
   factor for RTPS inflation.
   """
   raise SystemExit(msg)

# horizontal covariance localization length scale in meters.
hcovlocal_scale = 1.e3*float(sys.argv[1])
# RTPS inflation coeff
covinflate = float(sys.argv[2])

# other parameters.
#div2_diff_efold=1800.
div2_diff_efold=1.e30
fix_totmass = True # if True, use a mass fixer to fix mass in each layer (area mean dz)
posterior_stats = False
nassim = 1600 # assimilation times to run
#nassim = 10
ntime_savestart = 600 # if savedata is not None, start saving data at this time
nanals = 20 # ensemble members
savedata = None # if not None, netcdf filename to save data.
#savedata = True # filename given by exptname env var
# nature run created using twolayer_naturerun.py.
filename_climo = 'twolayerpe_N64_6hrly_symjet.nc' # file name for forecast model climo
# perfect model
#filename_truth = filename_climo
filename_truth = 'twolayerpe_N128_6hrly_symjet_nskip2.nc' # file name for forecast model climo
dzmin = 10. # min layer thickness allowed

profile = False # turn on profiling?

print('# filename_modelclimo=%s' % filename_climo)
print('# filename_truth=%s' % filename_truth)

exptname = os.getenv('exptname','test')
# get envar to set number of multiprocessing jobs for LETKF and ensemble forecast
n_jobs = int(os.getenv('N_JOBS','0'))
threads = 1

read_restart = False
debug_model = False # run perfect model ensemble, check to see that error=zero with no DA
precision = 'float32'

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
oberrstdev_wind = 1.   # wind ob error in meters per second
#oberrstdev_zsfc = np.sqrt(1.e30) # surface height ob error in meters
#oberrstdev_wind = np.sqrt(1.e30) # don't assimilate winds

ft = Fouriert(N,L,threads=threads,precision=precision) # create Fourier transform object

model = TwoLayer(ft,dt,zmid=zmid,ztop=ztop,tdrag1=tdrag[0],tdrag2=tdrag[1],tdiab=tdiab,div2_diff_efold=div2_diff_efold,\
umax=umax,theta1=theta1,theta2=theta2,diff_efold=diff_efold)
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
    if len(sys.argv) > 3:
        nt = int(sys.argv[3])
    else:
        nt = -1
    ncinit = Dataset('%s.nc' % exptname, mode='r', format='NETCDF4_CLASSIC')
    ncinit.set_auto_mask(False)
    uens[:] = ncinit.variables['u_b'][nt,...]
    vens[:] = ncinit.variables['v_b'][nt,...]
    dzens[:] = ncinit.variables['dz_b'][nt,...]
    tstart = ncinit.variables['t'][nt]
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

print("# hcovlocal=%g covinf=%s nanals=%s" %\
         (hcovlocal_scale/1000.,covinflate,nanals))

# each ob time nobs ob locations are randomly sampled (without
# replacement) from the model grid
#nobs = Nt**2 # observe full grid
nobs = Nt**2//16
#nobs = 1

# nature run
nc_truth = Dataset(filename_truth)
u_truth = nc_truth.variables['u']
v_truth = nc_truth.variables['v']
dz_truth = nc_truth.variables['dz']

# set up arrays for obs and localization function
print('# random network nobs = %s' % nobs)
oberrvar = np.ones(6*nobs,dtype)
oberrvar[0:4*nobs] = oberrstdev_wind**2*oberrvar[0:4*nobs]
oberrvar[4*nobs:5*nobs] = oberrstdev_zsfc**2*oberrvar[4*nobs:5*nobs]
oberrvar[5*nobs:] = oberrstdev_zmid**2*oberrvar[5*nobs:]
if nobs == 1:
    oberrvar[0:5:nobs]=1.e30
oberrstd = np.sqrt(oberrvar)

nobstot = 6*nobs
obs = np.empty(nobstot,dtype)
covlocal1 = np.empty(Nt**2,dtype)
covlocal1_tmp = np.empty((nobs,Nt**2),dtype)
covlocal_tmp = np.empty((nobstot,Nt**2),dtype)

obtimes = nc_truth.variables['t'][:]
if read_restart:
    timeslist = obtimes.tolist()
    ntstart = timeslist.index(tstart)
    print('# restarting from %s.nc ntstart = %s' % (exptname,ntstart))
else:
    ntstart = 0
assim_interval = obtimes[1]-obtimes[0]
assim_timesteps = int(np.round(assim_interval/model.dt))
print('# assim_interval = %s assim_timesteps = %s' % (assim_interval,assim_timesteps))
print('# div2_diff_efold = %s' % div2_diff_efold)
print('# oberrzsfc=%s oberrzmid=%s oberrwind=%s' % (oberrstdev_zsfc,oberrstdev_zmid,oberrstdev_wind))

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

def enstoctl_mean(uens,vens,dzens):
    Nt = uens.shape[-1]
    xens = np.empty((6,Nt**2),uens.dtype)
    # update u,v
    xens[0:2,:] = uens.reshape(2,Nt**2)
    xens[2:4,:] = vens.reshape(2,Nt**2)
    xens[4:6,:] = dzens.reshape(2,Nt**2)
    return xens

def enstoctl(uens,vens,dzens,sqrtcovlocal):
    nanals = uens.shape[0]
    Nt = uens.shape[-1]
    xens = np.empty((nanals,6,Nt**2),uens.dtype)
    # update u,v
    xens[:,0:2,:] = uens[:].reshape(nanals,2,Nt**2)
    xens[:,2:4,:] = vens[:].reshape(nanals,2,Nt**2)
    xens[:,4:6,:] = dzens[:].reshape(nanals,2,Nt**2)
    # modulated ensemble
    neig = sqrtcovlocal.shape[0]
    nanals2 = nanals*neig
    if neig > 1:
        upert = modens(uens,sqrtcovlocal)
        vpert = modens(vens,sqrtcovlocal)
        dzpert = modens(dzens,sqrtcovlocal)
        xens2 = np.empty((nanals2,6,Nt**2),uens.dtype)
        xens2[:,0:2,:] = upert[:].reshape(nanals2,2,Nt**2) 
        xens2[:,2:4,:] = vpert[:].reshape(nanals2,2,Nt**2)
        xens2[:,4:6,:] = dzpert[:].reshape(nanals2,2,Nt**2)
    else:
        xens2 = xens
    return xens, xens2

def ctltoens(xens):
    nanals = xens.shape[0]
    Nt = int(np.sqrt(xens.shape[-1]))
    uens = np.empty((nanals,2,Nt,Nt),xens.dtype)
    vens = np.empty((nanals,2,Nt,Nt),xens.dtype)
    dzens = np.empty((nanals,2,Nt,Nt),xens.dtype)
    uens[:] = xens[:,0:2,:].reshape(nanals,2,Nt,Nt)
    vens[:] = xens[:,2:4,:].reshape(nanals,2,Nt,Nt)
    dzens[:] = xens[:,4:6,:].reshape(nanals,2,Nt,Nt)
    return uens,vens,dzens

def inflation(xprime_a,xprime_b,covinflate):
    nanals = xprime_a.shape[0]
    asprd = (xprime_a**2).sum(axis=0)/(nanals-1)
    fsprd = (xprime_b**2).sum(axis=0)/(nanals-1)
    # relaxation to prior stdev (Whitaker & Hamill 2012)
    asprd = np.sqrt(asprd); fsprd = np.sqrt(fsprd)
    inflation_factor = 1.+covinflate*(fsprd-asprd)/asprd
    return xprime_a*inflation_factor

# specify localization matrix and square root
# (two-dimensional periodic domain with Nt x Nt)
t1 = time.time()
noloc = False
if noloc:
    neig = 1
    sqrtcovlocal = np.ones((neig,Nt,Nt),np.float32)
else:    
    covlocal = np.zeros((Nt**2,Nt**2),np.float32)
    xx = model.x.reshape(Nt**2)
    yy = model.y.reshape(Nt**2)
    for n in range(Nt**2):
        dist = cartdist(xx[n],yy[n],x,y,nc_climo.L,nc_climo.L)
        covlocal[:,n] = (gaspcohn(dist/hcovlocal_scale)).reshape(Nt**2)
    evals, evecs = eigh(covlocal,driver='evd')
    neig = 1
    for nn in range(1,Nt**2):
        percentvar = evals[-nn:].sum()/evals.sum()
        if percentvar > 0.98:
            neig = nn
            break
    print('#neig = ',neig)
    evecs_norm = (evecs*np.sqrt(evals/percentvar)).T
    sqrtcovlocal = evecs_norm[-neig:,:].reshape(neig,Nt,Nt)
    t2 = time.time()
    if profile: print('time for sqrtcovlocal calc',t2-t1)
nanals2 = neig*nanals

masstend_diag = 0.
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
    elif nobs == 1:
        zintensmean = model.ztop-dzens[1].mean(axis=0)
        zintprime = model.ztop-dzens[1]-zintensmean
        zintsprd  = (zintprime**2).sum(axis=0)/(nanals-1)
        #print(zintsprd.min(), zintsprd.max())
        #jmax,imax=np.unravel_index(zintsprd.argmax(), zintsprd.shape)
        #print(zintsprd[jmax,imax])
        indxob = np.atleast_1d(zintsprd.argmax())
        print(indxob)
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

    # compute forward operator.
    # hxens is ensemble in observation space.
    hxens = gethofx(uens,vens,ztop-dzens.sum(axis=1),ztop-dzens[:,1,...],indxob,nanals,nobs)
    hxensmean = hxens.mean(axis=0)
    hxprime_b = hxens - hxensmean

    # save data.
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

    uensmean_b = uens.mean(axis=0); vensmean_b = vens.mean(axis=0)
    upert_b = uens-uensmean_b; vpert_b=vens-vensmean_b
    dzensmean_b = dzens.mean(axis=0); dzpert_b = dzens-dzensmean_b

    if neig > 1:
        upert_bm = modens(upert_b,sqrtcovlocal)
        vpert_bm = modens(vpert_b,sqrtcovlocal)
        dzpert_bm = modens(dzpert_b,sqrtcovlocal)
    else:
        upert_bm=upert_b; vpert_bm=vpert_b; dzpert_bm=dzpert_b
    hxprime_bm = gethofx(upert_bm,vpert_bm,-dzpert_bm.sum(axis=1),-dzpert_bm[:,1,...],indxob,nanals2,nobs)
    #hxprime_bm = gethofx(uensmean_b+upert_bm,vensmean_b+vpert_bm,ztop-(dzensmean_b+dzpert_bm).sum(axis=1),ztop-(dzensmean_b+dzpert_bm)[:,1,...],indxob,nanals2,nobs)-hxensmean

    xensmean_b = enstoctl_mean(uensmean_b,vensmean_b,dzensmean_b)
    xprime_b,xprime_bm = enstoctl(upert_b,vpert_b,dzpert_b,sqrtcovlocal)

    if not debug_model: 
        dep = obs-hxensmean
        t1 = time.time()
        wts_ensmean,wts_ensperts = getkf(hxprime_b,hxprime_bm,oberrvar,dep)
        xensmean_a = np.empty_like(xensmean_b)
        xprime_a = np.empty_like(xprime_b)
        for k in range(xprime_bm.shape[1]):
            # increments constructed from weighted modulated ensemble member prior perts.
            xensmean_a[k,:] = xensmean_b[k,:] + np.dot(wts_ensmean,xprime_bm[:,k,:])
            xprime_a[:,k,:] = xprime_b[:,k,:] + np.dot(wts_ensperts,xprime_bm[:,k,:])
        t2 = time.time()
        if profile: print('time in getkf',t2-t1)
        xprime_a = xprime_a - xprime_a.mean(axis=0)
        xprime_a = inflation(xprime_a,xprime_b,covinflate)
        t2 = time.time()
        if profile: print('cpu time for EnKF update',t2-t1)
        xens = xprime_a + xensmean_a 

    # back to 3d state vector
    uens,vens,dzens = ctltoens(xens)
    if nobs==1: # single ob, set read_restart=True
    #if 1:
        uensmean_a = uens.mean(axis=0)
        vensmean_a = vens.mean(axis=0)
        dzensmean_a = dzens.mean(axis=0)
        zsensmean_b = dzensmean_b.sum(axis=0)
        zsensmean_a = dzensmean_a.sum(axis=0)
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        dzplot = (dzensmean_a - dzensmean_b)[1]
        zsplot = zsensmean_a-zsensmean_b
        uplot = (uensmean_a - uensmean_b)[1]
        vplot = (vensmean_a - vensmean_b)[1]
        print('uplot',uplot.min(), uplot.max())
        print('vplot',vplot.min(), vplot.max())
        print('dzplot',dzplot.min(), dzplot.max())
        plt.figure()
        vmin = -25; vmax = 25
        plt.imshow(vplot,cmap=plt.cm.bwr,vmin=vmin,vmax=vmax,interpolation="nearest")
        plt.colorbar()
        plt.title('v increment')
        plt.savefig('vinc.png')
        plt.figure()
        vmin = -25; vmax = 25
        plt.imshow(uplot,cmap=plt.cm.bwr,vmin=vmin,vmax=vmax,interpolation="nearest")
        plt.colorbar()
        plt.title('u increment')
        plt.savefig('uinc.png')
        plt.figure()
        vmin = -2500; vmax = 2500
        plt.imshow(dzplot,cmap=plt.cm.bwr,vmin=vmin,vmax=vmax,interpolation="nearest")
        plt.colorbar()
        plt.title('h increment')
        plt.savefig('hinc.png')
        raise SystemExit

    np.clip(dzens,a_min=dzmin,a_max=model.ztop-dzmin, out=dzens)
    if fix_totmass:
        for nmem in range(nanals):
            dzens[nmem][0] = dzens[nmem][0] - dzens[nmem][0].mean() + model.zmid
            dzens[nmem][1] = dzens[nmem][1] - dzens[nmem][1].mean() + model.ztop - model.zmid

    # posterior stats
    if posterior_stats:
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
