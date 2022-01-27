import numpy as np
from pyspharm import Spharmt

# two-layer baroclinic primitive equation model of
# Zou., X. A., A. Barcilon, I. M. Navon, J. S. Whitaker, and D. G. Cacuci,
# 1993: An adjoint sensitivity study of blocking in a two-layer isentropic
# model. Mon. Wea. Rev., 121, 2834-2857.
# doi: http://dx.doi.org/10.1175/1520-0493(1993)121<2833:AASSOB>2.0.CO;2
# see also https://journals.ametsoc.org/view/journals/mwre/133/11/mwr3020.1.xml

class TwoLayer(object):

    def __init__(self,sp,dt,theta1=280,theta2=310,grav=9.80616,omega=7.292e-5,cp=1004,\
                 zmid=5.e3,ztop=15.e3,efold=3600,ndiss=8,tdrag=4,tdiab=20,umax=25,jetexp=2,hmax=2.e3):
        # setup model parameters
        self.theta1 = theta1 # lower layer pot. temp.
        self.theta2 = theta2 # upper layer pot. temp.
        self.delth = theta2-theta1 # difference in potential temp between layers
        self.grav = grav # gravity
        self.hmax = hmax # orographic amplitude
        self.omega = omega # rotation rate
        self.cp = cp # Specific Heat of Dry Air at Constant Pressure,
        self.zmid = zmid # resting depth of lower layer (m)
        self.ztop = ztop # resting depth of both layers (m)
        self.umax = umax # equilibrium jet strength
        self.jetexp = jetexp # equlibrium jet width parameter
        # efolding time scale for hyperdiffusion at shortest wavenumber
        self.efold = efold
        self.ndiss = ndiss # order of hyperdiffusion (2 for laplacian)
        self.sp = sp # Spharmt instance
        self.ntrunc = sp.ntrunc # triangular truncation wavenumber
        self.dt = dt # time step (secs)
        self.tdiab = tdiab*86400. # lower layer drag timescale
        self.tdrag = tdrag*86400. # interface relaxation timescale
        # create lat/lon arrays
        delta = 2.*np.pi/sp.nlons
        lons1d = np.arange(-np.pi,np.pi,delta)
        lons,lats = np.meshgrid(lons1d,sp.lats)
        self.lons = lons
        self.lats = lats
        mu = np.sin(lats)
        self.f = 2.*omega*mu[np.newaxis,:,:] # coriolis
        # create laplacian operator and its inverse.
        indxn = sp.degree.astype(np.float)[np.newaxis,:]
        totwavenum = indxn*(indxn+1.0)
        self.lap = -totwavenum/sp.rsphere**2
        self.ilap = np.zeros(self.lap.shape, np.float)
        self.ilap[:,1:] = 1./self.lap[:,1:]
        # hyperdiffusion operator
        self.hyperdiff = -(1./efold)*(totwavenum/totwavenum[0,-1])**(ndiss/2)
        # initialize orography
        self.orog = 4.*hmax*(mu**2 - mu**4)*np.sin(2.*lons)
        # set equilibrium layer thicknes profile.
        self._interface_profile(umax,jetexp)
        self.t = 0

    def _interface_profile(self,umax,jetexp):
        ug = np.zeros((2,self.sp.nlats,self.sp.nlons),np.float)
        vg = np.zeros((2,self.sp.nlats,self.sp.nlons),np.float)
        ug[1,:,:] = umax*np.sin(2.*self.lats)**jetexp
        vrtspec, divspec = self.sp.getvrtdivspec(ug,vg)
        lyrthkspec = self.nlbalance(vrtspec)
        self.lyrthkref = self.sp.spectogrd(lyrthkspec)
        self.uref = ug
        if self.lyrthkref.min() < 0:
            raise ValueError('negative layer thickness! adjust equilibrium jet parameter')

    def nlbalance(self,vrtspec):
        # solve nonlinear balance eqn to get layer thickness given vorticity.
        divspec = np.zeros(vrtspec.shape, vrtspec.dtype)
        lyrthkspec = np.zeros(vrtspec.shape, vrtspec.dtype)
        vrtg = self.sp.spectogrd(vrtspec)
        ug,vg = self.sp.getuv(vrtspec,divspec)
        tmpg1 = ug*(vrtg+self.f); tmpg2 = vg*(vrtg+self.f)
        tmpspec1, tmpspec2 = self.sp.getvrtdivspec(tmpg1,tmpg2)
        tmpspec2 = self.sp.grdtospec(0.5*(ug**2+vg**2))
        mspec = self.ilap*tmpspec1 - tmpspec2
        mgrid = self.sp.spectogrd(mspec)
        lyrthkspec[0,:] =\
        (mspec[0,:]-self.sp.grdtospec(self.grav*self.orog))/self.theta1
        lyrthkspec[1,:] = (mspec[1,:]-mspec[0,:])/self.delth
        lyrthkspec[0,:] = lyrthkspec[0,:] - lyrthkspec[1,:]
        exnftop = self.cp - (self.grav*self.ztop/self.theta1)
        exnfmid = self.cp - (self.grav*self.zmid/self.theta1)
        lyrthkspec[0,0] = self.cp - exnfmid
        lyrthkspec[1,0] = exnfmid - exnftop
        lyrthkspec = (self.theta1/self.grav)*lyrthkspec # convert from exner function to height units (m)
        return lyrthkspec

    def gettend(self,vrtspec,divspec,lyrthkspec):
        # compute tendencies.
        # first, transform fields from spectral space to grid space.
        vrtg = self.sp.spectogrd(vrtspec)
        ug,vg = self.sp.getuv(vrtspec,divspec)
        lyrthkg = self.sp.spectogrd(lyrthkspec)
        self.u = ug; self.v = vg
        self.vrt = vrtg; self.lyrthk = lyrthkg
        if self.tdiab < 1.e10:
            totthk = lyrthkg.sum(axis=0)
            thtadot = self.delth*(self.lyrthkref[1,:,:] - lyrthkg[1,:,:])/\
                                (self.tdiab*totthk)
        # horizontal vorticity flux
        tmpg1 = ug*(vrtg+self.f); tmpg2 = vg*(vrtg+self.f)
        # add lower layer drag contribution
        if self.tdrag < 1.e10:
            tmpg1[0,:,:] += vg[0,:,:]/self.tdrag
            tmpg2[0,:,:] += -ug[0,:,:]/self.tdrag
        # add diabatic momentum flux contribution
        if self.tdiab < 1.e10:
            tmpg1 += 0.5*(vg[1,:,:]-vg[0,:,:])[np.newaxis,:,:]*\
            thtadot[np.newaxis,:,:]*totthk[np.newaxis,:,:]/(self.delth*lyrthkg)
            tmpg2 += -0.5*(ug[1,:,:]-ug[0,:,:])[np.newaxis,:,:]*\
            thtadot[np.newaxis,:,:]*totthk[np.newaxis,:,:]/(self.delth*lyrthkg)
        # compute vort flux contributions to vorticity and divergence tend.
        ddivdtspec, dvrtdtspec = self.sp.getvrtdivspec(tmpg1,tmpg2)
        dvrtdtspec *= -1
        # vorticity hyperdiffusion
        dvrtdtspec += self.hyperdiff*vrtspec
        # horizontal mass flux contribution to continuity
        tmpg1 = ug*lyrthkg; tmpg2 = vg*lyrthkg
        tmpspec, dlyrthkdtspec = self.sp.getvrtdivspec(tmpg1,tmpg2)
        dlyrthkdtspec *= -1
        # diabatic mass flux contribution to continuity
        if self.tdiab < 1.e10:
            tmpspec = self.sp.grdtospec(thtadot*totthk/self.delth)
            dlyrthkdtspec[0,:] += -tmpspec; dlyrthkdtspec[1,:] += tmpspec
        # pressure gradient force contribution to divergence tend (includes
        # orography).
        mstrm = np.empty((2,self.sp.nlats,self.sp.nlons),np.float)
        mstrm[0,:,:] = self.grav*(self.orog + lyrthkg[0,:,:] + lyrthkg[1,:,:])
        mstrm[1,:,:] = mstrm[0,:,:] +\
        (self.grav*self.delth/self.theta1)*lyrthkg[1,:,:]
        ddivdtspec += -self.lap*self.sp.grdtospec(mstrm+0.5*(ug**2+vg**2))
        # divergence hyperdiffusion
        ddivdtspec += self.hyperdiff*divspec
        return dvrtdtspec,ddivdtspec,dlyrthkdtspec

    def rk4step(self,vrtspec,divspec,lyrthkspec):
        # update state using 4th order runge-kutta
        dt = self.dt
        k1vrt,k1div,k1thk = \
        self.gettend(vrtspec,divspec,lyrthkspec)
        k2vrt,k2div,k2thk = \
        self.gettend(vrtspec+0.5*dt*k1vrt,divspec+0.5*dt*k1div,lyrthkspec+0.5*dt*k1thk)
        k3vrt,k3div,k3thk = \
        self.gettend(vrtspec+0.5*dt*k2vrt,divspec+0.5*dt*k2div,lyrthkspec+0.5*dt*k2thk)
        k4vrt,k4div,k4thk = \
        self.gettend(vrtspec+dt*k3vrt,divspec+dt*k3div,lyrthkspec+dt*k3thk)
        vrtspec += dt*(k1vrt+2.*k2vrt+2.*k3vrt+k4vrt)/6.
        divspec += dt*(k1div+2.*k2div+2.*k3div+k4div)/6.
        lyrthkspec += dt*(k1thk+2.*k2thk+2.*k3thk+k4thk)/6.
        self.t += dt
        return vrtspec,divspec,lyrthkspec

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('qt5agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # grid, time step info
    nlons = 96  # number of longitudes
    nlats = nlons//2
    ntrunc = nlons//3 # spectral truncation (for alias-free computations)
    #gridtype = 'regular'
    gridtype = 'gaussian'
    dt = 1200 # time step in seconds
    itmax = 100*(86400/dt) # integration length in days

    # create spherical harmonic instance.
    rsphere = 6.37122e6 # earth radius
    sp = Spharmt(nlons,nlats,ntrunc,rsphere,gridtype='gaussian')

    # create model instance using default parameters.
    model=TwoLayer(sp,dt)

    # vort, div initial conditions
    psipert = np.zeros((2,sp.nlats,sp.nlons),np.float)
    psipert[1,:,:] = 5.e6*np.sin((model.lons-np.pi))**12*np.sin(2.*model.lats)**12
    psipert = np.where(model.lons[np.newaxis,:,:] > 0., 0, psipert)
    ug = np.zeros((2,sp.nlats,sp.nlons),np.float)
    vg = np.zeros((2,sp.nlats,sp.nlons),np.float)
    ug[1,:,:] = model.umax*np.sin(2.*model.lats)**model.jetexp
    vrtspec, divspec = sp.getvrtdivspec(ug,vg)
    vrtspec = vrtspec + model.lap*sp.grdtospec(psipert)
    vrtg = sp.spectogrd(vrtspec)
    lyrthkspec = model.nlbalance(vrtspec)
    lyrthkg = sp.spectogrd(lyrthkspec)
    if lyrthkg.min() < 0:
        raise ValueError('negative layer thickness! adjust jet parameters')

    # animate pv
    fig = plt.figure(figsize=(16,8))
    vrtspec, divspec, lyrthkspec = model.rk4step(vrtspec, divspec, lyrthkspec)
    pv = (0.5*model.zmid/model.omega)*(model.vrt + model.f)/model.lyrthk
    vmin = -2.; vmax = 2.
    ax = fig.add_subplot(111); ax.axis('off')
    plt.tight_layout()
    im=ax.imshow(pv[1],cmap=plt.cm.nipy_spectral,vmin=vmin,vmax=vmax,interpolation="nearest")
    txt=ax.text(0.5,0.95,'Upper Layer PV day %10.2f' % float(model.t/86400.),\
                ha='center',color='k',fontsize=18,transform=ax.transAxes)

    model.t = 0 # reset clock
    nout = int(3.*3600./model.dt) # plot interval
    def updatefig(*args):
        global vrtspec, divspec, lyrthkspec
        for n in range(nout):
            vrtspec, divspec, lyrthkspec = model.rk4step(vrtspec, divspec,\
                    lyrthkspec)
        pv = (0.5*model.zmid/model.omega)*(model.vrt + model.f)/model.lyrthk
        im.set_data(pv[1])
        txt.set_text('Upper Layer PV day %10.2f' % float(model.t/86400.))
        return im,txt,

    ani = animation.FuncAnimation(fig,updatefig,interval=0,blit=False)
    plt.show()

