import numpy as np
from netCDF4 import Dataset
import sys, time, os
from twolayer import TwoLayer, run_model
from pyfft import Fouriert
from enkf_utils import cartdist,letkfwts_compute,gaspcohn
from joblib import Parallel, delayed

# LETKF cycling for two-layer pe turbulence model with interface height obs.
# horizontal localization (no vertical).
# Relaxation to prior spread
# inflation, or Hodyss and Campbell inflation.
# random observing network.
# Balanced/unbalanced partitioning.

if len(sys.argv) == 1:
   msg="""
python twolayerpe_letkf.py hcovlocal_scale <covinflate1 covinflate2>
   hcovlocal_scale = horizontal localization scale in km
   no vertical localization.
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

# other parameters.
diff_efold = None # use diffusion from climo file
#div2_diff_efold=1800.
div2_diff_efold=1.e30
fix_totmass = True # if True, use a mass fixer to fix mass in each layer (area mean dz)
baldiv = False # compute balanced divergence (if False, assign div to unbalanced part)
posterior_stats = False
nassim = 1600 # assimilation times to run
nanals = 20 # ensemble members
savedata = None # if not None, netcdf filename to save data.
#savedata = True # filename given by exptname env var
oberrstdev_zmid = 100.  # interface height ob error in meters
#oberrstdev_zsfc = 10. # surface height ob error in meters
#oberrstdev_wind = np.sqrt(2.) # wind ob error in meters per second
oberrstdev_zsfc = 1.e30 # surface height ob error in meters
oberrstdev_wind = 1.e30 # don't assimilate winds
# nature run created using twolayer_naturerun.py.
filename_climo = 'twolayerpe_N64_6hrly.nc' # file name for forecast model climo
# perfect model
#filename_truth = filename_climo
filename_truth = 'twolayerpe_N128_6hrly_nskip2.nc' # file name for forecast model climo
nobal = False # no balanced/unbalanced partitioning of control vector
linbal = False # use linear (geostrophic) balance instead of nonlinear (gradient) balance.
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
jetexp = nc_climo.jetexp
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

ft = Fouriert(N,L,threads=threads,precision=precision) # create Fourier transform object

model = TwoLayer(ft,dt,zmid=zmid,ztop=ztop,tdrag=tdrag,tdiab=tdiab,div2_diff_efold=div2_diff_efold,\
umax=umax,jetexp=jetexp,theta1=theta1,theta2=theta2,diff_efold=diff_efold)
if debug_model:
   print('N,Nt,L=',N,Nt,L)
   print('theta1,theta2=',theta1,theta2)
   print('zmid,ztop=',zmid,ztop)
   print('tdrag,tdiag=',tdrag/86400,tdiab/86400.)
   print('umax,jetexp=',umax,jetexp)
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

print("# hcovlocal=%g nobal=%s linbal=%s baldiv=%s covinf1=%s covinf2=%s nanals=%s" %\
     (hcovlocal_scale/1000.,nobal,linbal,baldiv,covinflate1,covinflate2,nanals))

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
print('# ntime,zmiderr,zmidsprd,v2err,v2sprd,zsfcerr,zsfcsprd,v1err,v1sprd,masstend_diag')

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
    nc.jetexp = jetexp
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
    m2ensmean = m2.mean(axis=0)
    zmiderr = (zmidensmean-zmid_truth)**2
    zmidprime = zmid-zmidensmean
    zmidsprd = (zmidprime**2).sum(axis=0)/(nanals-1)
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
    ke_errav = np.sqrt(ke_err.mean())
    ke_sprdav = np.sqrt(ke_sprd.mean())
    zmid_errav = np.sqrt(zmiderr.mean())
    zmid_sprdav = np.sqrt(zmidsprd.mean())
    m2_errav = np.sqrt(m2err.mean())
    m2_sprdav = np.sqrt(m2sprd.mean())
    zsfc_errav = np.sqrt(zsfcerr.mean())
    zsfc_sprdav = np.sqrt(zsfcsprd.mean())
    return vecwind1_errav,vecwind1_sprdav,vecwind2_errav,vecwind2_sprdav,zsfc_errav,zsfc_sprdav,zmid_errav,zmid_sprdav,m2_errav,m2_sprdav,ke_errav,ke_sprdav

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

def balens(model,uens,vens,dzens,linbal=False,baldiv=False,nitermax=1000,divguess=True,relax=0.015,eps=1.e-4,verbose=False):
    if not baldiv:
        # balanced div assumed zero
        divguess=None
    nanals = uens.shape[0]
    uens_bal = np.empty(uens.shape, uens.dtype)
    vens_bal = np.empty(uens.shape, uens.dtype)
    dzens_bal = np.empty(uens.shape, uens.dtype)
    for nmem in range(nanals):
        if verbose: print('ens member ',nmem)
        vrtspec, divspec = model.ft.getvrtdivspec(uens[nmem],vens[nmem])
        if divguess==True:
            div = model.ft.spectogrd(divspec) # use calculated div as initial guess
            nodiv=False
        elif divguess==False:
            div = True # use zeros as initial guess
            nodiv=False
        else:
            div = False # don't compute balanced div
            nodiv=True
        dz1mean = dzens[nmem,...][0].mean()
        dz2mean = dzens[nmem,...][1].mean()
        dzbal, divbal = model.nlbalance(vrtspec,linbal=linbal,div=div,dz1mean=dz1mean,dz2mean=dz2mean,\
                        nitermax=nitermax,relax=relax,eps=eps,verbose=verbose)
        if type(div) == bool and div == False:
            # no balanced divergence (much faster)
            divspec = np.zeros(vrtspec.shape, vrtspec.dtype)
        else:
            divspec = model.ft.grdtospec(divbal)
        uens_bal[nmem], vens_bal[nmem] = model.ft.getuv(vrtspec,divspec)
        dzens_bal[nmem] = dzbal
    return uens_bal,vens_bal,dzens_bal

def balmem(N,L,dt,umem,vmem,dzmem,linbal=False,baldiv=False,divguess=True,nitermax=1000,relax=0.015,eps=1.e-4,verbose=False,\
           theta1=300,theta2=320,f=1.e-4,div2_diff_efold=1.e30,\
           zmid=5.e3,ztop=10.e3,diff_efold=6.*3600.,diff_order=8,tdrag=4*86400,tdiab=20*86400,umax=12.5,jetexp=2):
    if not baldiv:
        # balanced div assumed zero
        divguess=None
    ft = Fouriert(N,L,threads=1)
    model=TwoLayer(ft,dt,theta1=theta1,theta2=theta2,f=f,div2_diff_efold=div2_diff_efold,\
    zmid=zmid,ztop=ztop,diff_efold=diff_efold,diff_order=diff_order,tdrag=tdrag,tdiab=tdiab,umax=umax,jetexp=jetexp)
    vrtspec, divspec = ft.getvrtdivspec(umem,vmem)
    if divguess==True:
        div = model.ft.spectogrd(divspec) # use calculated div as initial guess
        nodiv = False
    elif divguess==False:
        div = True # use zeros as initial guess
        nodiv = False
    else:
        div = False # don't compute balanced div
        nodiv = True
    dz1mean = dzmem[0].mean()
    dz2mean = dzmem[1].mean()
    dzbal, divbal = model.nlbalance(vrtspec,linbal=linbal,div=div,dz1mean=dz1mean,dz2mean=dz2mean,\
                    nitermax=nitermax,relax=relax,eps=eps,verbose=verbose)
    if type(div) == bool and div == False:
        # no balanced divergence (much faster)
        divspec = np.zeros(vrtspec.shape, vrtspec.dtype)
    else:
        divspec = model.ft.grdtospec(divbal)
    ubal,vbal = model.ft.getuv(vrtspec,divspec)
    return ubal,vbal,dzbal

def enstoctl(model,uens,vens,dzens,nobal=False,linbal=False,baldiv=False):
    nanals = uens.shape[0]
    if nobal:
        xens = np.empty((nanals,6,Nt**2),dtype)
        xens[:,0:2,:] = uens[:].reshape(nanals,2,model.ft.Nt**2)
        xens[:,2:4,:] = vens[:].reshape(nanals,2,model.ft.Nt**2)
        xens[:,4:6,:] = dzens[:].reshape(nanals,2,model.ft.Nt**2)
    else:
        if n_jobs == 0:
            uens_bal,vens_bal,dzens_bal = balens(model,uens,vens,dzens,linbal=linbal,baldiv=baldiv)
        else:
            results = Parallel(n_jobs=n_jobs)(delayed(balmem)(N,L,dt,uens[nanal],vens[nanal],dzens[nanal],linbal=linbal,baldiv=baldiv,divguess=True,theta1=model.theta1,theta2=model.theta2,zmid=model.zmid,ztop=model.ztop,diff_efold=model.diff_efold,diff_order=model.diff_order,tdrag=model.tdrag,tdiab=model.tdiab,umax=model.umax,jetexp=model.jetexp,div2_diff_efold=model.div2_diff_efold) for nanal in range(nanals))
            uens_bal = np.empty(uens.shape, uens.dtype); vens_bal = np.empty(vens.shape, vens.dtype)
            dzens_bal = np.empty(dzens.shape, dzens.dtype)
            for nanal in range(nanals):
                uens_bal[nanal],vens_bal[nanal],dzens_bal[nanal] = results[nanal]
        uens_unbal = uens-uens_bal
        vens_unbal = vens-vens_bal
        dzens_unbal = dzens-dzens_bal
        xens = np.empty((nanals,10,Nt**2),dtype)
        xens[:,0:2,:] = uens_bal[:].reshape(nanals,2,model.ft.Nt**2) # carries signal of balanced part
        xens[:,2:4,:] = vens_bal[:].reshape(nanals,2,model.ft.Nt**2)
        xens[:,4:6,:] = uens_unbal[:].reshape(nanals,2,model.ft.Nt**2)
        xens[:,6:8,:] = vens_unbal[:].reshape(nanals,2,model.ft.Nt**2)
        xens[:,8:10,:] = dzens_unbal[:].reshape(nanals,2,model.ft.Nt**2)
    return xens

def ctltoens(model,xens,nobal=False,linbal=False,baldiv=False):
    nanals = xens.shape[0]
    if nobal:
        uens = np.empty((nanals,2,Nt,Nt),dtype)
        vens = np.empty((nanals,2,Nt,Nt),dtype)
        dzens = np.empty((nanals,2,Nt,Nt),dtype)
        uens[:] = xens[:,0:2,:].reshape(nanals,2,model.ft.Nt,model.ft.Nt)
        vens[:] = xens[:,2:4,:].reshape(nanals,2,model.ft.Nt,model.ft.Nt)
        dzens[:] = xens[:,4:6,:].reshape(nanals,2,model.ft.Nt,model.ft.Nt)
    else:
        uens = np.empty((nanals,2,Nt,Nt),dtype)
        vens = np.empty((nanals,2,Nt,Nt),dtype)
        dzens = np.empty((nanals,2,Nt,Nt),dtype)
        uens_bal = np.empty((nanals,2,Nt,Nt),dtype)
        vens_bal = np.empty((nanals,2,Nt,Nt),dtype)
        dzens_bal = np.empty((nanals,2,Nt,Nt),dtype)
        dzens_bal[:,0,...] = model.zmid
        dzens_bal[:,1,...] = model.ztop - model.zmid
        uens_bal[:] = xens[:,0:2,:].reshape(nanals,2,model.ft.Nt,model.ft.Nt)
        vens_bal[:] = xens[:,2:4,:].reshape(nanals,2,model.ft.Nt,model.ft.Nt)
        uens[:] = xens[:,4:6,:].reshape(nanals,2,model.ft.Nt,model.ft.Nt)
        vens[:] = xens[:,6:8,:].reshape(nanals,2,model.ft.Nt,model.ft.Nt)
        dzens[:] = xens[:,8:10,:].reshape(nanals,2,model.ft.Nt,model.ft.Nt)
        if n_jobs == 0:
            uens_bal,vens_bal,dzens_bal = balens(model,uens_bal,vens_bal,dzens_bal,linbal=linbal,baldiv=baldiv)
        else:
            results = Parallel(n_jobs=n_jobs)(delayed(balmem)(N,L,dt,uens_bal[nanal],vens_bal[nanal],dzens_bal[nanal],linbal=linbal,baldiv=baldiv,divguess=True,theta1=model.theta1,theta2=model.theta2,zmid=model.zmid,ztop=model.ztop,diff_efold=model.diff_efold,diff_order=model.diff_order,tdrag=model.tdrag,tdiab=model.tdiab,umax=model.umax,jetexp=model.jetexp,div2_diff_efold=model.div2_diff_efold) for nanal in range(nanals))
            for nanal in range(nanals):
                uens_bal[nanal],vens_bal[nanal],dzens_bal[nanal] = results[nanal]
        uens += uens_bal
        vens += vens_bal
        dzens += dzens_bal
    return uens,vens,dzens

def inflation(xens,xensmean_b,fsprd,covinflate1,covinflate2):
    nanals = xens.shape[0]
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
    #inflation_factor = np.where(inflation_factor < 1, 1. inflation_factor)
    #print(inflation_factor.min(), inflation_factor.max(), inflation_factor.mean())
    xprime = xprime*inflation_factor
    xens = xprime + xensmean_a
    return xens

masstend_diag = 0.
for ntime in range(nassim):

    # turn off balanced divergence and unbalanced update in spinup
    if ntime < 100:
        baldiv2=False
    else:
        baldiv2=baldiv

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
    covlocal_tmp[nobs:2*nobs,:] = covlocal1_tmp
    covlocal_tmp[2*nobs:3*nobs,:] = covlocal1_tmp
    covlocal_tmp[3*nobs:4*nobs,:] = covlocal1_tmp
    covlocal_tmp[4*nobs:5*nobs,:] = covlocal1_tmp
    covlocal_tmp[5*nobs:,:] = covlocal1_tmp

    # compute forward operator.
    # hxens is ensemble in observation space.
    hxens = gethofx(uens,vens,ztop-dzens.sum(axis=1),ztop-dzens[:,1,...],indxob,nanals,nobs)

    # calculate LETKF weights
    wts = letkfwts_compute(hxens,obs,oberrvar,covlocal_tmp,n_jobs)

    if savedata is not None:
        u_t[ntime] = u_truth[ntime+ntstart]
        u_b[ntime,:,:,:] = uens
        v_t[ntime] = v_truth[ntime+ntstart]
        v_b[ntime,:,:,:] = vens
        dz_t[ntime] = dz_truth[ntime+ntstart]
        dz_b[ntime,:,:,:] = dzens
        obsu1[ntime] = obs[0:nobs]
        obsv1[ntime] = obs[nobs:2*nobs]
        obsu2[ntime] = obs[2*nobs:3*nobs]
        obsu2[ntime] = obs[3*nobs:4*nobs]
        obszfc[ntime] = obs[4*nobs:5*nobs]
        obszmid[ntime] = obs[5*nobs:]
        x_obs[ntime] = xob
        y_obs[ntime] = yob

    # prior stats.
    vecwind1_errav,vecwind1_sprdav,vecwind2_errav,vecwind2_sprdav,\
    zsfc_errav,zsfc_sprdav,zmid_errav,zmid_sprdav,m2_errav,m2_sprdav,ke_errav,ke_sprdav=\
    getspreaderr(model,uens,vens,dzens,\
    u_truth[ntime+ntstart],v_truth[ntime+ntstart],dz_truth[ntime+ntstart],ztop)
    totmass = ((dzens[:,0,...]+dzens[:,1,...]).mean(axis=0)).mean()/1000.
    print("%s %g %g %g %g %g %g %g %g %g %g %g %g  %g %g" %\
    (ntime+ntstart,zmid_errav,zmid_sprdav,m2_errav,m2_sprdav,vecwind2_errav,vecwind2_sprdav,\
     zsfc_errav,zsfc_sprdav,vecwind1_errav,vecwind1_sprdav,ke_errav,ke_sprdav,masstend_diag,totmass))

    # EnKF update for balanced part.
    xens = enstoctl(model,uens,vens,dzens,nobal=nobal,linbal=linbal,baldiv=baldiv2)
    xensmean_b = xens.mean(axis=0)
    xprime = xens-xensmean_b
    fsprd = (xprime**2).sum(axis=0)/(nanals-1)

    if not debug_model: 
        # update state vector using letkf weights
        #for k in range(4): # only update balanced u,v
        for k in range(xens.shape[1]):
            for n in range(model.ft.Nt**2):
                xens[:, k, n] = xensmean_b[k,n] + np.dot(wts[n].T, xprime[:, k, n])
        t2 = time.time()
        if profile: print('cpu time for EnKF update',t2-t1)
        xens = inflation(xens,xensmean_b,fsprd,covinflate1,covinflate2)

    uens,vens,dzens = ctltoens(model,xens,nobal=nobal,linbal=linbal,baldiv=baldiv2)

    # make sure there is no negative layer thickness in analysis
    np.clip(dzens,a_min=dzmin,a_max=model.ztop-dzmin, out=dzens)

    if fix_totmass: # not needed
        for nmem in range(nanals):
            dzens[nmem][0] = dzens[nmem][0] - dzens[nmem][0].mean() + model.zmid
            dzens[nmem][1] = dzens[nmem][1] - dzens[nmem][1].mean() + model.ztop - model.zmid

    # posterior stats
    if posterior_stats:
        vecwind1_errav,vecwind1_sprdav,vecwind2_errav,vecwind2_sprdav,\
        zsfc_errav,zsfc_sprdav,zmid_errav,zmid_sprdav,m2_errav,m2_sprdav,ke_errav,ke_sprdav=\
        getspreaderr(model,uens,vens,dzens,\
        u_truth[ntime+ntstart],v_truth[ntime+ntstart],dz_truth[ntime+ntstart],ztop)
        totmass = ((dzens[:,0,...]+dzens[:,1,...]).mean(axis=0)).mean()/1000.
        print("%s %g %g %g %g %g %g %g %g %g %g %g %g %g %g" %\
        (ntime+ntstart,zmid_errav,zmid_sprdav,m2_errav,m2_sprdav,vecwind2_errav,vecwind2_sprdav,\
         zsfc_errav,zsfc_sprdav,vecwind1_errav,vecwind1_sprdav,ke_errav,ke_sprdav,masstend_diag,totmass))

    # save data.
    if savedata is not None:
        u_a[ntime,:,:,:] = uens
        v_a[ntime,:,:,:] = vens
        dz_a[ntime,:,:,:] = dzens
        tvar[ntime] = obtimes[ntime+ntstart]
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
        results = Parallel(n_jobs=n_jobs)(delayed(run_model)(uens[nanal],vens[nanal],dzens[nanal],N,L,dt,assim_timesteps,theta1=theta1,theta2=theta2,zmid=zmid,ztop=ztop,diff_efold=diff_efold,diff_order=diff_order,tdrag=tdrag,tdiab=tdiab,umax=umax,jetexp=jetexp,div2_diff_efold=div2_diff_efold) for nanal in range(nanals))
        masstend_diag=0.
        for nanal in range(nanals):
            uens[nanal],vens[nanal],dzens[nanal],mtend = results[nanal]
            masstend_diag+=mtend/nanals
    model.t = tstart + dt*assim_timesteps
    t2 = time.time()
    if profile: print('cpu time for ens forecast',t2-t1)

if savedata: nc.close()
