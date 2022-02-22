import numpy as np
from netCDF4 import Dataset
import sys, time, os
from twolayer import TwoLayer, run_model
from pyfft import Fouriert
from enkf_utils import cartdist,letkf_update,serial_update,gaspcohn
from nlbal_utils import pvinvert
from joblib import Parallel, delayed

# EnKF cycling for two-layer pe turbulence model with interface height obs.
# horizontal localization (no vertical).
# Relaxation to prior spread
# inflation, or Hodyss and Campbell inflation.
# random observing network.
# Options for serial EnSRF or LETKF.

if len(sys.argv) == 1:
   msg="""
python twolayerpe_enkf_bal.py hcovlocal_scale hcovlocal_scale_unbal  <covinflate1 covinflate2>
   hcovlocal_scale = horizontal localization scale in km for balanced part
   hcovlocal_scale_unbal = horizontal localization scale in km for unbalanced part
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
# if hcovlocal_scale_unbal=1, unbalanced part cycled but not updated.
# if hcovlocal_scale_unbal=0, unbalanced part of analysis set to zero.
# if hcovlocal_scale_unbal>1, unbalanced part updated assuming bal and unbal parts uncorrelated.
hcovlocal_scale_unbal = 1.e3*float(sys.argv[2])

# optional inflation parameters:
# (covinflate2 <= 0 for RTPS inflation
# (http://journals.ametsoc.org/doi/10.1175/MWR-D-11-00276.1),
# otherwise use Hodyss et al inflation
# (http://journals.ametsoc.org/doi/abs/10.1175/MWR-D-15-0329.1)
if len(sys.argv) == 4:
    covinflate1 = float(sys.argv[3])
    covinflate2 = -1
elif len(sys.argv) == 5:
    covinflate1 = float(sys.argv[3])
    covinflate2 = float(sys.argv[4])
else:
    covinflate1 = 1.
    covinflate2 = 1.

exptname = os.getenv('exptname','test')
# get envar to set number of multiprocessing jobs for LETKF and ensemble forecast
n_jobs = int(os.getenv('N_JOBS','0'))
threads = 1

diff_efold = None # use diffusion from climo file

profile = False # turn on profiling?

use_letkf = True # if False, use serial EnSRF
baldiv = True # compute balanced divergence (if False, assign div to unbalanced part)
ivar = 0 # 0 for u,v update, 1 for pv
if ivar == 0:
    nlevs_update = 4
else:
    nlevs_update = 2
read_restart = False
debug_model = False # run perfect model ensemble, check to see that error=zero with no DA
posterior_stats = False
precision = 'float32'
savedata = None # if not None, netcdf filename to save data.
#savedata = True # filename given by exptname env var
nanals = 20 # ensemble members
nassim = 800 # assimilation times to run

oberrstdev_zmid = 100.  # interface height ob error in meters
#oberrstdev_wind = np.sqrt(2.) # wind ob error in meters per second
oberrstdev_wind = 1.e30 # don't assimilate winds

# nature run created using twolayer_naturerun.py.
filename_climo = 'twolayerpe_N64_6hrly_sp.nc' # file name for forecast model climo
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

model = TwoLayer(ft,dt,zmid=zmid,ztop=ztop,tdrag=tdrag,tdiab=tdiab,\
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
    ncinit = Dataset('%s_restart.nc' % exptname, mode='r', format='NETCDF4_CLASSIC')
    ncinit.set_auto_mask(False)
    uens[:] = ncinit.variables['v_b'][-1,...]
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

print("# hcovlocal=%g hcovlocal_unbal=%g baldiv=%s  use_letkf=%s covinf1=%s covinf2=%s nanals=%s" %\
     (hcovlocal_scale/1000.,hcovlocal_scale_unbal/1000.,baldiv,use_letkf,covinflate1,covinflate2,nanals))

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
oberrvar = np.ones(5*nobs,dtype)
oberrvar[0:4*nobs] = oberrstdev_wind*oberrvar[0:4*nobs]
oberrvar[4*nobs:] = oberrstdev_zmid*oberrvar[4*nobs:]

obs = np.empty(5*nobs,dtype)
covlocal1 = np.empty(Nt**2,dtype)
covlocal1u = np.empty(Nt**2,dtype)
covlocal1_tmp = np.empty((nobs,Nt**2),dtype)
covlocal1u_tmp = np.empty((nobs,Nt**2),dtype)
covlocal_tmp = np.empty((5*nobs,Nt**2),dtype)
xens = np.empty((nanals,6,Nt**2),dtype)
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
print('# assim_interval = %s assim_timesteps = %s' % (assim_interval,assim_timesteps))
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
def gethofx(uens,vens,zmidens,indxob,nanals,nobs):
    hxens = np.empty((nanals,5*nobs),dtype)
    for nanal in range(nanals):
        hxens[nanal,0:nobs] = uens[nanal,0,...].ravel()[indxob] # interface height obs
        hxens[nanal,nobs:2*nobs] = vens[nanal,0,...].ravel()[indxob] # interface height obs
        hxens[nanal,2*nobs:3*nobs] = uens[nanal,1,...].ravel()[indxob] # interface height obs
        hxens[nanal,3*nobs:4*nobs] = vens[nanal,1,...].ravel()[indxob] # interface height obs
        hxens[nanal,4*nobs:] = zmidens[nanal,...].ravel()[indxob] # interface height obs
    return hxens

if baldiv:
    nodiv = False
else:
    nodiv = True

def balens(model,uens,vens,dzens,nodiv=True,nitermax=1000,relax=0.015,eps=1.e-4,verbose=False):
    nanals = uens.shape[0]
    uens_bal = np.empty(uens.shape, uens.dtype)
    vens_bal = np.empty(uens.shape, uens.dtype)
    dzens_bal = np.empty(uens.shape, uens.dtype)
    for nmem in range(nanals):
        vrtspec, divspec = model.ft.getvrtdivspec(uens[nmem],vens[nmem])
        vrt = model.ft.spectogrd(vrtspec)
        pv = (vrt + model.f)/dzens[nmem]
        dzbal, vrtbal, divbal = pvinvert(model,pv,dzin=dzens[nmem],nodiv=nodiv,\
                                nitermax=nitermax,relax=relax,eps=eps,verbose=verbose)
        #dz1mean = dzens[nmem,0,...].mean()
        #dz2mean = dzens[nmem,1,...].mean()
        #dzbal, vrtbal, divbal = pvinvert(model,pv,dzin=None,dz1mean=dz1mean,dz2mean=dz2mean,nodiv=nodiv,\
        #                        nitermax=nitermax,relax=relax,eps=eps,verbose=verbose)
        vrtspec = model.ft.grdtospec(vrtbal)
        if nodiv:
            # no balanced divergence.
            divspec = np.zeros(vrtspec.shape, vrtspec.dtype)
        else:
            divspec = model.ft.grdtospec(divbal)
        uens_bal[nmem], vens_bal[nmem] = model.ft.getuv(vrtspec,divspec)
        dzens_bal[nmem] = dzbal
    return uens_bal,vens_bal,dzens_bal

def enstoctl(model,uens,vens,dzens,ivar=0):
    xens = np.empty((nanals,6,Nt**2),dtype)
    if ivar==0:
        # update u,v,dz
        xens[:,0:2,:] = uens[:].reshape(nanals,2,model.ft.Nt**2)
        xens[:,2:4,:] = vens[:].reshape(nanals,2,model.ft.Nt**2)
    elif ivar==1:
        # update pv,div,dz
        for nmem in range(nanals):
            vrtspec,divspec = model.ft.getvrtdivspec(uens[nmem],vens[nmem])
            vrt = model.ft.spectogrd(vrtspec); div = model.ft.spectogrd(divspec)
            pv = (vrt + model.f)/dzens[nmem]
            xens[nmem,0:2,:] = pv[:].reshape(2,model.ft.Nt**2)
            xens[nmem,2:4,:] = div[:].reshape(2,model.ft.Nt**2)
    else: 
        raise ValueError('ivar myst be 0 or 1')
    xens[:,4:6,:] = dzens[:].reshape(nanals,2,model.ft.Nt**2)
    return xens

def ctltoens(model,xens,ivar=0,nodiv=True,
             nitermax=1000,relax=0.015,eps=1.e-4,verbose=False):
    uens = np.empty((nanals,2,Nt,Nt),dtype)
    vens = np.empty((nanals,2,Nt,Nt),dtype)
    dzens = np.empty((nanals,2,Nt,Nt),dtype)
    if ivar == 0:
        uens[:] = xens[:,0:2,:].reshape(nanals,2,model.ft.Nt,model.ft.Nt)
        vens[:] = xens[:,2:4,:].reshape(nanals,2,model.ft.Nt,model.ft.Nt)
        dzens[:] = xens[:,4:6,:].reshape(nanals,2,model.ft.Nt,model.ft.Nt)
    elif ivar == 1:
        for nmem in range(nanals):
            pv = xens[nmem,0:2,:].reshape(2,model.ft.Nt,model.ft.Nt)
            dz = xens[nmem,4:6,:].reshape(2,model.ft.Nt,model.ft.Nt) # equal to background, not updated
            # invert pv balanced u,v,dz
            dzbal, vrtbal, divbal = pvinvert(model,pv,dzin=dz,nodiv=nodiv,\
                                    nitermax=nitermax,relax=relax,eps=eps,verbose=verbose)
            vrtspec = model.ft.grdtospec(vrtbal); divspec = model.ft.grdtospec(divbal)
            uens[nmem], vens[nmem] = model.ft.getuv(vrtspec,divspec)
            dzens[nmem] = dzbal
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
    obs[4*nobs:] = ztop - dz_truth[ntime+ntstart,1,:,:].ravel()[indxob] +\
                   rsobs.normal(scale=oberrstdev_zmid,size=nobs) 
    xob = x.ravel()[indxob]
    yob = y.ravel()[indxob]

    # compute covariance localization function for each ob
    for nob in range(nobs):
        dist = cartdist(xob[nob],yob[nob],x,y,nc_climo.L,nc_climo.L)
        covlocal1 = gaspcohn(dist/hcovlocal_scale)
        covlocal1_tmp[nob] = covlocal1.ravel()
        covlocal1u = gaspcohn(dist/hcovlocal_scale_unbal)
        covlocal1u_tmp[nob] = covlocal1u.ravel()
        dist = cartdist(xob[nob],yob[nob],xob,yob,nc_climo.L,nc_climo.L)
        if not use_letkf: obcovlocal1[nob] = gaspcohn(dist/hcovlocal_scale)
    covlocal_tmp[nobs:2*nobs,:] = covlocal1_tmp
    covlocal_tmp[2*nobs:3*nobs,:] = covlocal1_tmp
    covlocal_tmp[3*nobs:4*nobs,:] = covlocal1_tmp
    covlocal_tmp[4*nobs:,:] = covlocal1_tmp
    if not use_letkf:
        obcovlocal = np.block(
            [
                [obcovlocal1,obcovlocal1,obcovlocal1,obcovlocal1,obcovlocal1],
                [obcovlocal1,obcovlocal1,obcovlocal1,obcovlocal1,obcovlocal1],
                [obcovlocal1,obcovlocal1,obcovlocal1,obcovlocal1,obcovlocal1],
                [obcovlocal1,obcovlocal1,obcovlocal1,obcovlocal1,obcovlocal1],
                [obcovlocal1,obcovlocal1,obcovlocal1,obcovlocal1,obcovlocal1]
            ]
        )

    # split background into balanced and unbalanced parts
    # update balanced and unbalanced parts separately
    # (assuming no cross-covariance)
    # impose balance constraint on balanced part of ens after the update
    uens_bal,vens_bal,dzens_bal = balens(model,uens,vens,dzens,nodiv=nodiv)
    uens_unbal = uens-uens_bal
    vens_unbal = vens-vens_bal
    dzens_unbal = dzens-dzens_bal

    # compute forward operator.
    # hxens is ensemble in observation space.
    hxens = gethofx(uens,vens,ztop-dzens[:,1,...],indxob,nanals,nobs)

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
        obszmid[ntime] = obs[4*nobs:]
        x_obs[ntime] = xob
        y_obs[ntime] = yob

    # prior stats.
    vecwind1_errav_b,vecwind1_sprdav_b,vecwind2_errav_b,vecwind2_sprdav_b,\
    zsfc_errav_b,zsfc_sprdav_b,zmid_errav_b,zmid_sprdav_b=getspreaderr(uens,vens,dzens,\
    u_truth[ntime+ntstart],v_truth[ntime+ntstart],dz_truth[ntime+ntstart],ztop)
    print("%s %g %g %g %g %g %g %g %g %g %g" %\
    (ntime+ntstart,zmid_errav_b,zmid_sprdav_b,vecwind2_errav_b,vecwind2_sprdav_b,\
     zsfc_errav_b,zsfc_sprdav_b,vecwind1_errav_b,vecwind1_sprdav_b,inflation_factor.mean(),masstend_diag))

    # EnKF update for balanced part.
    xens = enstoctl(model,uens_bal,vens_bal,dzens_bal,ivar=ivar)
    xensmean_b = xens.mean(axis=0)
    xprime = xens-xensmean_b
    fsprd = (xprime**2).sum(axis=0)/(nanals-1)

    # update state vector with serial filter or letkf.
    if not debug_model: 
        if use_letkf:
            # only update first four fields (dzens_bal will be inferred from balanced u,v)
            xens = letkf_update(xens,hxens,obs,oberrvar,covlocal_tmp,n_jobs,nlevs_update=nlevs_update) 
        else:
            xens = serial_update(xens,hxens,obs,oberrvar,covlocal_tmp,obcovlocal)
        t2 = time.time()
        if profile: print('cpu time for EnKF update',t2-t1)
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
        # make sure inflation not less than 1
        #inflation_factor = np.where(inflation_factor < 1, 1, inflation_factor)
        xprime = xprime*inflation_factor
        xens = xprime + xensmean_a

    uens_bal,vens_bal,dzens_bal = ctltoens(model,xens,ivar=ivar,nodiv=nodiv)
    # balance 'balanced' analysis ensemble for u,v,dz control vector
    # (done inside ctltoens for pv control vector)
    if ivar==0:
        uens_bal,vens_bal,dzens_bal = balens(model,uens_bal,vens_bal,dzens_bal,nodiv=nodiv)

    if hcovlocal_scale_unbal > 1001: # otherwise don't update unbalanced part.
        # EnKF update for unbalanced part.
        xens = enstoctl(model,uens_unbal,vens_unbal,dzens_unbal,ivar=0)
        xensmean_b = xens.mean(axis=0)
        xprime = xens-xensmean_b
        fsprd = (xprime**2).sum(axis=0)/(nanals-1)

        # (different localization for unbalanced part)
        covlocal_tmp[nobs:2*nobs,:] = covlocal1u_tmp
        covlocal_tmp[2*nobs:3*nobs,:] = covlocal1u_tmp
        covlocal_tmp[3*nobs:4*nobs,:] = covlocal1u_tmp
        covlocal_tmp[4*nobs:,:] = covlocal1u_tmp

        # update state vector with serial filter or letkf.
        if not debug_model: 
            if use_letkf:
                xens = letkf_update(xens,hxens,obs,oberrvar,covlocal_tmp,n_jobs)
            else:
                xens = serial_update(xens,hxens,obs,oberrvar,covlocal_tmp,obcovlocal)
            t2 = time.time()
            if profile: print('cpu time for EnKF update',t2-t1)
            xensmean_a = xens.mean(axis=0)
            xprime = xens-xensmean_a
            asprd = (xprime**2).sum(axis=0)/(nanals-1)
            if covinflate2 < 0:
                # relaxation to prior stdev (Whitaker & Hamill 2012)
                asprd = np.sqrt(asprd); fsprd = np.sqrt(fsprd)
                inflation_factoru = 1.+covinflate1*(fsprd-asprd)/asprd
            else:
                # Hodyss et al 2016 inflation (covinflate1=covinflate2=1 works well in perfect
                # model, linear gaussian scenario)
                # inflation = asprd + (asprd/fsprd)**2((fsprd/nanals)+2*inc**2/(nanals-1))
                inc = xensmean_a - xensmean_b
                inflation_factoru = covinflate1*asprd + \
                (asprd/fsprd)**2*((fsprd/nanals) + covinflate2*(2.*inc**2/(nanals-1)))
                inflation_factoru = np.sqrt(inflation_factoru/asprd)
            # make sure inflation not less than 1
            #inflation_factor = np.where(inflation_factor < 1, 1, inflation_factor)
            xprime = xprime*inflation_factoru
            xens = xprime + xensmean_a

        uens_unbal,vens_unbal,dzens_unbal = ctltoens(model,xens,ivar=0)

    if hcovlocal_scale_unbal == 0:
        # the unbalanced component of the analysis is set to zero
        uens = uens_bal
        vens = vens_bal
        dzens = dzens_bal
        #dzens = dzens_bal + dzens_unbal # really only need this (and only for ens mean)
    else:
        # note if hcovlocal_scale specified as 1 on command line, the unbalanced
        # component is not updated (background is just cycled)
        uens = uens_bal + uens_unbal
        vens = vens_bal + vens_unbal
        dzens = dzens_bal + dzens_unbal

    # posterior stats
    if posterior_stats:
        vecwind1_errav_a,vecwind1_sprdav_a,vecwind2_errav_a,vecwind2_sprdav_a,\
        zsfc_errav_a,zsfc_sprdav_a,zmid_errav_a,zmid_sprdav_a=getspreaderr(uens,vens,dzens,\
        u_truth[ntime+ntstart],v_truth[ntime+ntstart],dz_truth[ntime+ntstart],ztop)
        print("%s %g %g %g %g %g %g %g %g" %\
        (ntime+ntstart,zmid_errav_a,zmid_sprdav_a,vecwind2_errav_a,vecwind2_sprdav_a,\
         zsfc_errav_a,zsfc_sprdav_a,vecwind1_errav_a,vecwind1_sprdav_a))

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
        results = Parallel(n_jobs=n_jobs)(delayed(run_model)(uens[nanal],vens[nanal],dzens[nanal],N,L,dt,assim_timesteps,theta1=theta1,theta2=theta2,zmid=zmid,ztop=ztop,diff_efold=diff_efold,diff_order=diff_order,tdrag=tdrag,tdiab=tdiab,umax=umax,jetexp=jetexp) for nanal in range(nanals))
        masstend_diag=0.
        for nanal in range(nanals):
            uens[nanal],vens[nanal],dzens[nanal],mtend = results[nanal]
            masstend_diag+=mtend/nanals
    model.t = tstart + dt*assim_timesteps
    t2 = time.time()
    if profile: print('cpu time for ens forecast',t2-t1)

if savedata: nc.close()
