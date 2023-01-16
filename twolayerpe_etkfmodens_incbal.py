import numpy as np
from netCDF4 import Dataset
import sys, time, os
from twolayer_bal import TwoLayer, run_model
from pyfft import Fouriert
from enkf_utils import cartdist,gaspcohn
from scipy.linalg import eigh, svd
from joblib import Parallel, delayed

def modens(enspert,sqrtcovlocal):
    nanals = enspert.shape[0]
    neig = sqrtcovlocal.shape[0]
    #nlevs = enspert.shape[1]
    #Nt = enspert.shape[2]
    #enspertm = np.empty((nanals*neig,nlevs,Nt,Nt),enspert.dtype)
    #for k in range(nlevs):
    #   nanal2 = 0
    #   for j in range(neig):
    #       for nanal in range(nanals):
    #           enspertm[nanal2,k,...] = enspert[nanal,k,...]*sqrtcovlocal[j,...]
    #           nanal2+=1
    enspertm = np.multiply(np.repeat(sqrtcovlocal[:,np.newaxis,:,:],nanals,axis=0),np.tile(enspert,(neig,1,1,1)))
    #print(sqrtcovlocal.min(), sqrtcovlocal.max(),(enspertm-enspertm2).min(), (enspertm-enspertm2).max())
    return enspertm

# LETKF cycling for two-layer pe turbulence model with interface height obs.
# horizontal localization (no vertical).
# Relaxation to prior spread inflatino.
# Balanced/unbalanced partitioning using incremental balance.

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
fix_totmass = False # if True, use a mass fixer to fix mass in each layer (area mean dz)
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
linbal = False # use linear (geostrophic) balance instead of nonlinear (gradient) balance.
dzmin = 10. # min layer thickness allowed
inflate_before=True # inflate before balance operator applied
baldiv = False # include balanced divergence
update_unbal = True # update unbalanced part

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

print("# hcovlocal=%g linbal=%s baldiv=%s covinf=%s nanals=%s" %\
         (hcovlocal_scale/1000.,linbal,baldiv,covinflate,nanals))

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
oberrvar[0:4*nobs] = oberrstdev_wind**2*oberrvar[0:4*nobs]
oberrvar[4*nobs:5*nobs] = oberrstdev_zsfc**2*oberrvar[4*nobs:5*nobs]
oberrvar[5*nobs:] = oberrstdev_zmid**2*oberrvar[5*nobs:]
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

def balpert(N,L,dt,upert,vpert,vrtspec_ensmean,divspec_ensmean,dz_ensmean,linbal=False,baldiv=baldiv,\
           theta1=300,theta2=320,f=1.e-4,div2_diff_efold=1.e30,\
           zmid=5.e3,ztop=10.e3,diff_efold=6.*3600.,diff_order=8,tdrag1=10*86400,tdrag2=10*86400,tdiab=20*86400,umax=8):
    ft = Fouriert(N,L,threads=1)
    model=TwoLayer(ft,dt,theta1=theta1,theta2=theta2,f=f,div2_diff_efold=div2_diff_efold,\
    zmid=zmid,ztop=ztop,diff_efold=diff_efold,diff_order=diff_order,tdrag1=tdrag[0],tdrag2=tdrag[1],tdiab=tdiab,umax=umax)
    vrtspec, divspec = ft.getvrtdivspec(upert,vpert)
    dzpert_bal,divpert_bal = model.nlbalinc(vrtspec_ensmean,divspec_ensmean,dz_ensmean,vrtspec,linbal=linbal,baldiv=baldiv)
    if baldiv:
        divspec = model.ft.grdtospec(divpert_bal)
    else:
        divspec = np.zeros_like(vrtspec)
    upert_bal,vpert_bal = model.ft.getuv(vrtspec,divspec)
    return upert_bal,vpert_bal,dzpert_bal

def balenspert(model,upert,vpert,vrtspec_ensmean,divspec_ensmean,dz_ensmean,linbal=linbal,baldiv=baldiv):
    upert_bal = np.empty_like(upert)
    vpert_bal = np.empty_like(vpert)
    dzpert_bal = np.empty_like(vpert)
    for nanal in range(nanals):
        vrtspec, divspec = ft.getvrtdivspec(upert[nanal],vpert[nanal])
        dzpert_bal[nanal],divpert_bal = model.nlbalinc(vrtspec_ensmean,divspec_ensmean,dz_ensmean,vrtspec,linbal=linbal,baldiv=baldiv)
        if baldiv:
            divspec = model.ft.grdtospec(divpert_bal)
        else:
            divspec = np.zeros_like(vrtspec)
        upert_bal[nanal],vpert_bal[nanal] = model.ft.getuv(vrtspec,divspec)
    return upert_bal, vpert_bal, dzpert_bal

def enstoctl(model,upert,vpert,dzpert,vrtspec_ensmean,divspec_ensmean,dz_ensmean,linbal=False,baldiv=baldiv):
    """upert,vpert,dzpert to upert_bal,vpert_bal,upert_unbal,vpert_unbal,dzunbal_pert"""
    nanals = upert.shape[0]
    if n_jobs == 0:
        upert_bal,vpert_bal,dzpert_bal = balenspert(model,upert,vpert,vrtspec_ensmean,divspec_ensmean,dz_ensmean,linbal=linbal,baldiv=baldiv)
    else:
        results = Parallel(n_jobs=n_jobs)(delayed(balpert)(N,L,dt,upert[nanal],vpert[nanal],vrtspec_ensmean,divspec_ensmean,dz_ensmean,linbal=linbal,baldiv=baldiv,theta1=model.theta1,theta2=model.theta2,zmid=model.zmid,ztop=model.ztop,diff_efold=model.diff_efold,diff_order=model.diff_order,tdrag1=model.tdrag[0],tdrag2=model.tdrag[1],tdiab=model.tdiab,umax=model.umax,div2_diff_efold=model.div2_diff_efold) for nanal in range(nanals))
        upert_bal = np.empty(upert.shape, upert.dtype); vpert_bal = np.empty(vpert.shape, vpert.dtype)
        dzpert_bal = np.empty(dzpert.shape, dzpert.dtype)
        for nanal in range(nanals):
            upert_bal[nanal],vpert_bal[nanal],dzpert_bal[nanal] = results[nanal]
    upert_unbal = upert-upert_bal
    vpert_unbal = vpert-vpert_bal
    dzpert_unbal = dzpert-dzpert_bal
    xens = np.empty((nanals,10,Nt**2),dtype)
    xens[:,0:2,:] = upert_bal[:].reshape(nanals,2,model.ft.Nt**2) # carries signal of balanced part
    xens[:,2:4,:] = vpert_bal[:].reshape(nanals,2,model.ft.Nt**2)
    xens[:,4:6,:] = upert_unbal[:].reshape(nanals,2,model.ft.Nt**2)
    xens[:,6:8,:] = vpert_unbal[:].reshape(nanals,2,model.ft.Nt**2)
    xens[:,8:10,:] = dzpert_unbal[:].reshape(nanals,2,model.ft.Nt**2)

    return xens

def enstoctl2(model,upert,vpert,dzpert,vrtspec_ensmean,divspec_ensmean,dz_ensmean,sqrtcovlocal,linbal=False,baldiv=baldiv):
    """upert,vpert,dzpert to upert_bal,vpert_bal,upert_unbal,vpert_unbal,dzunbal_pert"""
    nanals = upert.shape[0]
    if n_jobs == 0:
        upert_bal,vpert_bal,dzpert_bal = balenspert(model,upert,vpert,vrtspec_ensmean,divspec_ensmean,dz_ensmean,linbal=linbal,baldiv=baldiv)
    else:
        results = Parallel(n_jobs=n_jobs)(delayed(balpert)(N,L,dt,upert[nanal],vpert[nanal],vrtspec_ensmean,divspec_ensmean,dz_ensmean,linbal=linbal,baldiv=baldiv,theta1=model.theta1,theta2=model.theta2,zmid=model.zmid,ztop=model.ztop,diff_efold=model.diff_efold,diff_order=model.diff_order,tdrag1=model.tdrag[0],tdrag2=model.tdrag[1],tdiab=model.tdiab,umax=model.umax,div2_diff_efold=model.div2_diff_efold) for nanal in range(nanals))
        upert_bal = np.empty(upert.shape, upert.dtype); vpert_bal = np.empty(vpert.shape, vpert.dtype)
        dzpert_bal = np.empty(dzpert.shape, dzpert.dtype)
        for nanal in range(nanals):
            upert_bal[nanal],vpert_bal[nanal],dzpert_bal[nanal] = results[nanal]
    upert_unbal = upert-upert_bal
    vpert_unbal = vpert-vpert_bal
    dzpert_unbal = dzpert-dzpert_bal
    xens = np.empty((nanals,10,Nt**2),dtype)
    xens[:,0:2,:] = upert_bal[:].reshape(nanals,2,model.ft.Nt**2) # carries signal of balanced part
    xens[:,2:4,:] = vpert_bal[:].reshape(nanals,2,model.ft.Nt**2)
    xens[:,4:6,:] = upert_unbal[:].reshape(nanals,2,model.ft.Nt**2)
    xens[:,6:8,:] = vpert_unbal[:].reshape(nanals,2,model.ft.Nt**2)
    xens[:,8:10,:] = dzpert_unbal[:].reshape(nanals,2,model.ft.Nt**2)

    # modulated ensemble
    neig = sqrtcovlocal.shape[0]
    nanals2 = nanals*neig
    upert_balbm = modens(upert_bal,sqrtcovlocal)
    vpert_balbm = modens(vpert_bal,sqrtcovlocal)
    upert_unbalbm = modens(upert_unbal,sqrtcovlocal)
    vpert_unbalbm = modens(vpert_unbal,sqrtcovlocal)
    dzpert_unbalbm = modens(dzpert_unbal,sqrtcovlocal)
    xens2 = np.empty((nanals2,10,Nt**2),dtype)
    xens2[:,0:2,:] = upert_balbm[:].reshape(nanals2,2,model.ft.Nt**2) 
    xens2[:,2:4,:] = vpert_balbm[:].reshape(nanals2,2,model.ft.Nt**2)
    xens2[:,4:6,:] = upert_unbalbm[:].reshape(nanals2,2,model.ft.Nt**2)
    xens2[:,6:8,:] = vpert_unbalbm[:].reshape(nanals2,2,model.ft.Nt**2)
    xens2[:,8:10,:] = dzpert_unbalbm[:].reshape(nanals2,2,model.ft.Nt**2)

    return xens, xens2

def ctltoens(model,xens,vrtspec_ensmean,divspec_ensmean,dz_ensmean,linbal=False,baldiv=baldiv):
    """upert_bal,vpert_bal,upert_unbal,vpert_unbal,dzunbal_pert to upert,vpert,dzpert"""
    nanals = xens.shape[0]
    upert_bal = np.empty((nanals,2,Nt,Nt),dtype)
    vpert_bal = np.empty((nanals,2,Nt,Nt),dtype)
    dzpert_bal = np.empty((nanals,2,Nt,Nt),dtype)
    upert_unbal = np.empty((nanals,2,Nt,Nt),dtype)
    vpert_unbal = np.empty((nanals,2,Nt,Nt),dtype)
    dzpert_unbal = np.empty((nanals,2,Nt,Nt),dtype)
    upert_bal[:] = xens[:,0:2,:].reshape(nanals,2,model.ft.Nt,model.ft.Nt)
    vpert_bal[:] = xens[:,2:4,:].reshape(nanals,2,model.ft.Nt,model.ft.Nt)
    upert_unbal[:] = xens[:,4:6,:].reshape(nanals,2,model.ft.Nt,model.ft.Nt)
    vpert_unbal[:] = xens[:,6:8,:].reshape(nanals,2,model.ft.Nt,model.ft.Nt)
    dzpert_unbal[:] = xens[:,8:10,:].reshape(nanals,2,model.ft.Nt,model.ft.Nt)
    if n_jobs == 0:
        upert_bal,vpert_bal,dzpert_bal = balenspert(model,upert_bal,vpert_bal,vrtspec_ensmean,divspec_ensmean,dz_ensmean,linbal=linbal,baldiv=baldiv)
    else:
        results = Parallel(n_jobs=n_jobs)(delayed(balpert)(N,L,dt,upert_bal[nanal],vpert_bal[nanal],vrtspec_ensmean,divspec_ensmean,dz_ensmean,linbal=linbal,baldiv=baldiv,theta1=model.theta1,theta2=model.theta2,zmid=model.zmid,ztop=model.ztop,diff_efold=model.diff_efold,diff_order=model.diff_order,tdrag1=model.tdrag[0],tdrag2=model.tdrag[1],tdiab=model.tdiab,umax=model.umax,div2_diff_efold=model.div2_diff_efold) for nanal in range(nanals))
        for nanal in range(nanals):
            upert_bal[nanal],vpert_bal[nanal],dzpert_bal[nanal] = results[nanal]
    upert = upert_bal + upert_unbal
    vpert = vpert_bal + vpert_unbal
    dzpert = dzpert_bal + dzpert_unbal
    return upert,vpert,dzpert

def inflation(xprime_a,xprime_b,covinflate):
    nanals = xprime_a.shape[0]
    asprd = (xprime_a**2).sum(axis=0)/(nanals-1)
    fsprd = (xprime_b**2).sum(axis=0)/(nanals-1)
    # relaxation to prior stdev (Whitaker & Hamill 2012)
    asprd = np.sqrt(asprd); fsprd = np.sqrt(fsprd)
    inflation_factor = 1.+covinflate*(fsprd-asprd)/asprd
    #inflation_factor = np.where(inflation_factor < 1, 1. inflation_factor)
    #print(inflation_factor.min(), inflation_factor.max(), inflation_factor.mean())
    return xprime_a*inflation_factor
    #return xprime_a*inflation_factor.mean() # constant inflation factor

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
    vrtspec_ensmean_b, divspec_ensmean_b = model.ft.getvrtdivspec(uensmean_b,vensmean_b)
    upert_b = uens-uensmean_b; vpert_b=vens-vensmean_b
    dzensmean_b = dzens.mean(axis=0); dzpert_b = dzens-dzensmean_b

    upert_bm = modens(upert_b,sqrtcovlocal)
    vpert_bm = modens(vpert_b,sqrtcovlocal)
    dzpert_bm = modens(dzpert_b,sqrtcovlocal)
    hxprime_bm = gethofx(upert_bm,vpert_bm,-dzpert_bm.sum(axis=1),-dzpert_bm[:,1,...],indxob,nanals2,nobs)

    # modulate after partitioning
    xprime_b,xprime_bm = enstoctl2(model,upert_b,vpert_b,dzpert_b,vrtspec_ensmean_b,divspec_ensmean_b,dzensmean_b,sqrtcovlocal,linbal=linbal,baldiv=baldiv2)

    # modulate, then partition
    #xprime_b = enstoctl(model,upert_b,vpert_b,dzpert_b,vrtspec_ensmean_b,divspec_ensmean_b,dzensmean_b,linbal=linbal,baldiv=baldiv2)
    #xprime_bm = enstoctl(model,upert_bm,vpert_bm,dzpert_bm,vrtspec_ensmean_b,divspec_ensmean_b,dzensmean_b,linbal=linbal,baldiv=baldiv2)

    if not debug_model: 
        xensmean_inc = np.zeros(xprime_b.shape[1:],xprime_b.dtype)
        xprime_a = xprime_b.copy()

        # getkf global solution with model-space localization
        # HZ^T = hxens * R**-1/2
        # compute eigenvectors/eigenvalues of HZ^T HZ (left SV)
        # (in Bishop paper HZ is nobs, nanals, here is it nanals, nobs)
        # normalize so dot product is covariance
        normfact = np.array(np.sqrt(nanals-1),dtype=np.float32)
        hxens = (hxprime_bm/oberrstd[np.newaxis,:])/normfact
        pa = np.dot(hxens,hxens.T)
        evals, evecs = eigh(pa, driver='evd')
        gamma_inv = np.zeros_like(evals)
        #evals = np.where(evals > np.finfo(evals.dtype).eps, evals, 0.)
        #gamma_inv = np.where(evals > np.finfo(evals.dtype).eps, 1./evals, 0.)
        for n in range(nanals2):
            if evals[n] > np.finfo(evals.dtype).eps:
               gamma_inv[n] = 1./evals[n]
            else:
               evals[n] = 0. 
        # gammapI used in calculation of posterior cov in ensemble space
        gammapI = evals+1.
        # create HZ^T R**-1/2 
        shxens = hxens/oberrstd[np.newaxis,:]
        # compute factor to multiply with model space ensemble perturbations
        # to compute analysis increment (for mean update), save in single precision.
        # This is the factor C (Gamma + I)**-1 C^T (HZ)^ T R**-1/2 (y - HXmean)
        # in Bishop paper (eqs 10-12).
        # pa = C (Gamma + I)**-1 C^T (analysis error cov in ensemble space)
        #    = matmul(evecs/gammapI,transpose(evecs))
        pa = np.dot(evecs/gammapI[np.newaxis,:],evecs.T)
        # wts_ensmean = C (Gamma + I)**-1 C^T (HZ)^ T R**-1/2 (y - HXmean)
        # (nanals, nanals) x (nanals,) = (nanals,)
        # do nanal=1,nanals
        #    swork1(nanal) = sum(shxens(nanal,:)*dep(:))
        # end do
        # do nanal=1,nanals
        #    wts_ensmean(nanal) = sum(pa(nanal,:)*swork1(:))/normfact
        # end do
        wts_ensmean = np.dot(pa, np.dot(shxens,obs-hxensmean))/normfact
        # compute factor to multiply with model space ensemble perturbations
        # to compute analysis increment (for perturbation update), save in single precision.
        # This is -C [ (I - (Gamma+I)**-1/2)*Gamma**-1 ] C^T (HZ)^T R**-1/2 HXprime
        # in Bishop paper (eqn 29).
        # For DEnKF factor is -0.5*C (Gamma + I)**-1 C^T (HZ)^ T R**-1/2 HXprime
        # = -0.5 Pa (HZ)^ T R**-1/2 HXprime (Pa already computed)
        # pa = C [ (I - (Gamma+I)**-1/2)*Gamma**-1 ] C^T
        # gammapI = sqrt(1.0/gammapI)
        # do nanal=1,nanals
        #    swork3(nanal,:) = &
        #    evecs(nanal,:)*(1.-gammapI(:))*gamma_inv(:)
        # enddo
        # pa = matmul(swork3,transpose(swork2))
        pa = np.dot(evecs*(1.-np.sqrt(1./gammapI[np.newaxis,:]))*gamma_inv[np.newaxis,:],evecs.T)
        #pa=0.5*pa for denkf
        # wts_ensperts = -C [ (I - (Gamma+I)**-1/2)*Gamma**-1 ] C^T (HZ)^T R**-1/2 HXprime
        # (nanals, nanals) x (nanals, nanals/eigv) = (nanals, nanals/neigv)
        # if denkf, wts_ensperts = -0.5 C (Gamma + I)**-1 C^T (HZ)^T R**-1/2 HXprime
        # swork2 = matmul(shxens,transpose(hxens_orig))
        #wts_ensperts = -matmul(pa, swork2)/normfact
        wts_ensperts = -np.dot(pa, np.dot(shxens,hxprime_b.T)).T/normfact
        #paens = pa/normfact**2 # posterior covariance in modulated ensemble space
        for k in range(xprime_b.shape[1]):
            # increments constructed from weighted modulated ensemble member prior perts.
            xensmean_inc[k, :] = np.dot(wts_ensmean,xprime_bm[:,k,:])
            #print(k,xensmean_inc[k,:].min(),xensmean_inc[k,:].max())
            xprime_a[:,k,:] = xprime_b[:,k,:] + np.dot(wts_ensperts,xprime_bm[:,k,:])
        xprime_a = xprime_a - xprime_a.mean(axis=0)

        t2 = time.time()
        if profile: print('cpu time for EnKF update',t2-t1)
        if inflate_before: xprime_a = inflation(xprime_a,xprime_b,covinflate)

    uensmean_balinc = np.empty_like(uensmean_b); vensmean_balinc = np.empty_like(vensmean_b)
    uensmean_unbalinc = np.empty_like(uensmean_b); vensmean_unbalinc = np.empty_like(vensmean_b)
    dzensmean_unbalinc = np.empty_like(uensmean_b)
    uensmean_balinc[:] = xensmean_inc[0:2,:].reshape(2,model.ft.Nt,model.ft.Nt)
    vensmean_balinc[:] = xensmean_inc[2:4,:].reshape(2,model.ft.Nt,model.ft.Nt)
    # get balanced ens mean increments from rotational wind increments
    uensmean_balinc,vensmean_balinc,dzensmean_balinc = \
    balpert(N,L,dt,uensmean_balinc,vensmean_balinc,vrtspec_ensmean_b,divspec_ensmean_b,dzensmean_b,linbal=linbal,baldiv=baldiv2)
    uensmean_unbalinc[:] = xensmean_inc[4:6,:].reshape(2,model.ft.Nt,model.ft.Nt)
    vensmean_unbalinc[:] = xensmean_inc[6:8,:].reshape(2,model.ft.Nt,model.ft.Nt)
    dzensmean_unbalinc[:] = xensmean_inc[8:10,:].reshape(2,model.ft.Nt,model.ft.Nt)
    # get total pertubation increments from unbalanced/balanced increments
    # reconstruct total analysis fields
    upertinc,vpertinc,dzpertinc = ctltoens(model,xprime_a-xprime_b,vrtspec_ensmean_b,divspec_ensmean_b,dzensmean_b,linbal=linbal,baldiv=baldiv2)
    if not inflate_before: 
        upert = upert_b + upertinc 
        vpert = vpert_b + vpertinc 
        dzpert = dzpert_b + dzpertinc 
        xens = np.empty((nanals,6,Nt**2),dtype)
        xens[:,0:2,:] = upert[:].reshape(nanals,2,model.ft.Nt**2)
        xens[:,2:4,:] = vpert[:].reshape(nanals,2,model.ft.Nt**2)
        xens[:,4:6,:] = dzpert[:].reshape(nanals,2,model.ft.Nt**2)
        xens_b = np.empty((nanals,6,Nt**2),dtype)
        xens_b[:,0:2,:] = upert_b[:].reshape(nanals,2,model.ft.Nt**2)
        xens_b[:,2:4,:] = vpert_b[:].reshape(nanals,2,model.ft.Nt**2)
        xens_b[:,4:6,:] = dzpert_b[:].reshape(nanals,2,model.ft.Nt**2)
        xprime_a = inflation(xens,xens_b,covinflate)
        upert[:] = xprime_a[:,0:2,:].reshape(nanals,2,model.ft.Nt,model.ft.Nt)
        vpert[:] = xprime_a[:,2:4,:].reshape(nanals,2,model.ft.Nt,model.ft.Nt)
        dzpert[:] = xprime_a[:,4:6,:].reshape(nanals,2,model.ft.Nt,model.ft.Nt)
        upertinc = upert - upert_b
        vpertinc = vpert - vpert_b
        dzpertinc = dzpert - dzpert_b
    uens = upert_b + uensmean_b + upertinc + uensmean_balinc + uensmean_unbalinc
    vens = vpert_b + vensmean_b + vpertinc + vensmean_balinc + vensmean_unbalinc
    dzens = dzpert_b + dzensmean_b + dzpertinc + dzensmean_balinc + dzensmean_unbalinc

    if nobs==1: # single ob, set read_restart=True
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
        #print(dzplot.min(), dzplot.max())
        print(zsplot.min(), zsplot.max())
        plt.figure()
        vmin = -0.1; vmax = 0.1
        plt.imshow(vplot,cmap=plt.cm.bwr,vmin=vmin,vmax=vmax,interpolation="nearest")
        plt.colorbar()
        plt.title('zs increment (analysis)')
        plt.savefig('zsinc.png')
        raise SystemExit

    # make sure there is no negative layer thickness in analysis
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