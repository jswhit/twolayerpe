import numpy as np
from pyfft import Fouriert

# f-plane version two-layer baroclinic primitive equation model
# same as dry version of mc2RSW model of Lambaerts 2011 (doi:10.1063/1.3582356)

# run on command line to generate an animation.  
# needs pyfftw and matplotlib with qt5agg backend for animation.

class TwoLayer(object):

    def __init__(self,ft,dt,theta1=300,theta2=320,f=1.e-4,\
                 zmid=5.e3,ztop=10.e3,diff_efold=6*3600.,diff_order=8,\
                 div2_diff_efold=1.e30,tdrag=4*86400,tdiab=20*86400,umax=12.5,jetexp=2):
        dtype = ft.precision
        # setup model parameters
        self.dtype = dtype
        self.theta1 = np.array(theta1,dtype) # lower layer pot. temp. (K)
        self.theta2 = np.array(theta2,dtype) # upper layer pot. temp. (K)
        self.delth = np.array(theta2-theta1,dtype) # difference in potential temp between layers
        self.grav = 9.80616 # gravity
        self.zmid = np.array(zmid,dtype) # resting depth of lower layer (m)
        self.ztop = np.array(ztop,dtype) # resting depth of both layers (m)
        self.umax = np.array(umax,dtype) # equilibrium jet strength
        self.jetexp = jetexp # jet width parameter (should be even, higher=narrower)
        self.ft = ft # Fouriert instance (defines domain size, grid resolution)
        self.dt = np.array(dt,dtype) # time step (secs)
        self.tdiab = np.array(tdiab,dtype) # lower layer drag timescale (secs)
        self.tdrag = np.array(tdrag,dtype) # interface relaxation timescale (secs)
        self.f = np.array(f,dtype) # coriolis parameter
        # hyperdiffusion parameters
        self.diff_order = np.array(diff_order, dtype)  # hyperdiffusion order
        self.diff_efold = np.array(diff_efold, dtype)  # hyperdiff time scale (secs)
        self.div2_diff_efold = np.array(div2_diff_efold, dtype) # extra laplacian div damping
        ktot = np.sqrt(self.ft.ksqlsq)
        pi = np.array(np.pi,dtype)  
        ktotcutoff = np.array(pi * self.ft.N / self.ft.L, dtype)
        # integrating factor for hyperdiffusion
        self.hyperdiff = -(1./self.diff_efold)*(ktot/ktotcutoff)**self.diff_order
        if div2_diff_efold < 1.e10:
            # extra laplacian diffusion of divergence to damp gravity waves
            self.divlapdiff = -(1./self.div2_diff_efold)*(ktot/ktotcutoff)**2
        else:
            self.divlapdiff = 0.
        x = np.arange(0, self.ft.L, self.ft.L / self.ft.Nt, dtype=dtype)
        y = np.arange(0, self.ft.L, self.ft.L / self.ft.Nt, dtype=dtype)
        x, y = np.meshgrid(x,y)
        self.x = x; self.y = y
        # set equilibrium layer thicknes profile.
        self._interface_profile(umax)
        self.t = 0.
        self.timesteps = 1
        self.masstendvar = 0

    def _interface_profile(self,umax):
        u = np.zeros((2,self.ft.Nt,self.ft.Nt),dtype=self.dtype)
        v = np.zeros((2,self.ft.Nt,self.ft.Nt),dtype=self.dtype)
        l = np.array(2*np.pi,self.dtype) / self.ft.L
        u[1] = umax*np.sin(l*self.y)*np.sin(l*self.y)**self.jetexp
        uspec = self.ft.grdtospec(u)
        vrtspec, divspec = self.ft.getvrtdivspec(u,v)
        u,v = self.ft.getuv(vrtspec,divspec)
        self.uref = u
        self.dzref, div = self.nlbalance(vrtspec)
        if self.dzref.min() < 0:
            raise ValueError('negative layer thickness! adjust equilibrium jet parameter')

    def nlbalance(self,vrtspec,div=False, dz1mean=None, dz2mean=None, nitermax=1000,\
                  relax=0.01, eps=1.e-2, verbose=False):
        """computes balanced layer thickness given vorticity (from nonlinear bal eqn)"""
        if dz1mean is None: 
            dz1mean = self.zmid
        if dz2mean is None:
            dz2mean = self.ztop - self.zmid

        psispec = self.ft.invlap*vrtspec

        psixx = self.ft.spectogrd(-self.ft.k**2*psispec)
        psiyy = self.ft.spectogrd(-self.ft.l**2*psispec)
        psixy = self.ft.spectogrd(self.ft.k*self.ft.l*psispec)

        tmpspec = self.f*vrtspec + 2.*self.ft.grdtospec(psixx*psiyy - psixy**2)
        mspec = self.ft.invlap*tmpspec
        dzspec = np.zeros(mspec.shape, mspec.dtype)
        dzspec[0,...] = mspec[0,...]/self.theta1
        dzspec[1,...] = (mspec[1,:]-mspec[0,...])/(self.theta2-self.theta1)
        dzspec[0,...] -= dzspec[1,...]
        dzspec = (self.theta1/self.grav)*dzspec # convert from exner function to height units (m)

        #divspec = np.zeros(vrtspec.shape, vrtspec.dtype)
        #dzspec = np.zeros(vrtspec.shape, vrtspec.dtype)
        #vrt = self.ft.spectogrd(vrtspec)
        #u,v = self.ft.getuv(vrtspec,divspec)
        #tmpg1 = u*(vrt+self.f); tmpg2 = v*(vrt+self.f)
        #tmpspec1, tmpspec2 = self.ft.getvrtdivspec(tmpg1,tmpg2)
        #tmpspec2 = self.ft.grdtospec(0.5*(u**2+v**2))
        #mspec = self.ft.invlap*tmpspec1 - tmpspec2
        #dzspec[0,...] = mspec[0,...]/self.theta1
        ##(mspec[0,...]-self.ft.grdtospec(self.grav*self.orog))/self.theta1 # with orography
        #dzspec[1,...] = (mspec[1,:]-mspec[0,...])/self.delth
        #dzspec[0,...] = dzspec[0,...] - dzspec[1,...]
        #dzspec = (self.theta1/self.grav)*dzspec # convert from exner function to height units (m)

        # set area mean in grid space to state of rest value
        dz = self.ft.spectogrd(dzspec)
        dz[0,...] = dz[0,...] - dz[0,...].mean() + dz1mean
        dz[1,...] = dz[1,...] - dz[1,...].mean() + dz2mean

        if type(div) == bool:
            if div == False: # don't compute balanced divergence
                div = np.zeros(dz.shape, dz.dtype)
                return dz,div
            elif div == True: # compute divergence, initialize iteration with div=0
                divspec = np.zeros(vrtspec.shape, vrtspec.dtype)
                div = np.zeros(dz.shape, dz.dtype)
        dzx,dzy = self.ft.getgrad(dz)
        urot = self.ft.spectogrd(-self.ft.il*psispec); vrot = self.ft.spectogrd(self.ft.ik*psispec)
        vrt = self.ft.spectogrd(vrtspec)

        def nlbalance_tend(dvrtdt):
            # solve tendency of nonlinear balance eqn to get layer thickness tendency
            # given vorticity tendency (psixx,psiyy and psixy already computed)
            dvrtspecdt = self.ft.grdtospec(dvrtdt)
            dpsispecdt = self.ft.invlap*dvrtspecdt
            dpsixxdt = self.ft.spectogrd(-self.ft.k**2*dpsispecdt)
            dpsiyydt = dvrtdt - dpsixxdt
            dpsixydt = self.ft.spectogrd(self.ft.k*self.ft.l*dpsispecdt)
            tmpspec = self.f*dvrtspecdt + 2.*self.ft.grdtospec(dpsixxdt*psiyy + psixx*dpsiyydt - 2*psixy*dpsixydt)
            mspec = self.ft.invlap*tmpspec
            dzspec = np.zeros(mspec.shape, mspec.dtype)
            dzspec[0,...] = mspec[0,...]/self.theta1
            dzspec[1,...] = (mspec[1,:]-mspec[0,...])/(self.theta2-self.theta1)
            dzspec[0,...] -= dzspec[1,...]
            dzspec = (self.theta1/self.grav)*dzspec # convert from exner function to height units (m)
            ddzdt = self.ft.spectogrd(dzspec)
            # remove area mean
            ddzdt[0,...] = ddzdt[0,...] - ddzdt[0,...].mean()
            ddzdt[1,...] = ddzdt[1,...] - ddzdt[1,...].mean()
            return ddzdt

        # get balanced divergence computed iterative algorithm
        # following appendix of https://doi.org/10.1175/1520-0469(1993)050<1519:ACOPAB>2.0.CO;2
        # start iteration with div=0
        converged=False
        for niter in range(nitermax):
            divspec = self.ft.grdtospec(div)
            chispec = self.ft.invlap*divspec
            udivspec = self.ft.ik*chispec; vdivspec = self.ft.il*chispec
            udiv = self.ft.spectogrd(udivspec); vdiv = self.ft.spectogrd(vdivspec)
            u = urot+udiv; v = vrot+vdiv
            # compute initial guess of vorticity tendency 
            # first, transform fields from spectral space to grid space.
            # diabatic mass flux due to interface relaxation.
            massflux = (self.dzref[1] - dz[1])/self.tdiab
            # horizontal vorticity flux
            tmp1 = u*(vrt+self.f); tmp2 = v*(vrt+self.f)
            # add lower layer drag contribution
            tmp1[0] += v[0]/self.tdrag
            tmp2[0] += -u[0]/self.tdrag
            # add diabatic momentum flux contribution
            # (this version averages vertical flux at top
            # and bottom of each layer)
            # same as 'improved' mc2RSW self (DOI: 10.1002/qj.3292)
            tmp1 += 0.5*(v[1]-v[0])*massflux/dz
            tmp2 -= 0.5*(u[1]-u[0])*massflux/dz
            # compute vort flux contributions to vorticity and divergence tend.
            ddivdtspec, dvrtdtspec = self.ft.getvrtdivspec(tmp1,tmp2)
            dvrtdtspec *= -1
            dvrtdtspec += self.hyperdiff*vrtspec
            dvrtdt = self.ft.spectogrd(dvrtdtspec)
            # infer layer thickness tendency from d/dt of balance eqn.
            ddzdt = nlbalance_tend(dvrtdt)
            # new estimate of divergence from continuity eqn
            tmp1[0] = massflux; tmp1[1] = -massflux
            divnew = -(1./dz)*(ddzdt + u*dzx + v*dzy - tmp1)
            divnew = divnew - divnew.mean() # remove area mean
            divdiff = divnew-div
            div = div + relax*divdiff
            divdiffmean = np.sqrt((divdiff**2).mean())
            divmean = np.sqrt((div**2).mean())
            if verbose: print(niter, divdiffmean, divdiffmean/divmean )
            if divdiffmean/divmean < eps:    
                converged = True
                break
        if not converged:
            raise RuntimeError('balanced divergence solution did not converge')

        # remove area mean div 
        div[0]=div[0]-div[0].mean()
        div[1]=div[1]-div[1].mean()

        return dz,div

    def gettend(self,vrtspec,divspec,dzspec):
        # compute tendencies.
        # first, transform fields from spectral space to grid space.
        vrt = self.ft.spectogrd(vrtspec)
        u,v = self.ft.getuv(vrtspec,divspec)
        dz = self.ft.spectogrd(dzspec)
        # diabatic mass flux due to interface relaxation.
        massflux = (self.dzref[1] - dz[1])/self.tdiab
        # horizontal vorticity flux
        tmpg1 = u*(vrt+self.f); tmpg2 = v*(vrt+self.f)
        # add lower layer drag contribution
        tmpg1[0] += v[0]/self.tdrag
        tmpg2[0] += -u[0]/self.tdrag
        # add diabatic momentum flux contribution
        # (this version averages vertical flux at top
        # and bottom of each layer)
        # same as 'improved' mc2RSW model (DOI: 10.1002/qj.3292)
        tmpg1 += 0.5*(v[1]-v[0])*massflux/dz
        tmpg2 -= 0.5*(u[1]-u[0])*massflux/dz
        # compute vort flux contributions to vorticity and divergence tend.
        ddivdtspec, dvrtdtspec = self.ft.getvrtdivspec(tmpg1,tmpg2)
        dvrtdtspec *= -1
        # horizontal mass flux contribution to continuity
        tmpg1 = u*dz; tmpg2 = v*dz
        tmpspec, ddzdtspec = self.ft.getvrtdivspec(tmpg1,tmpg2)
        ddzdtspec *= -1
        # diabatic mass flux contribution to continuity
        tmpspec = self.ft.grdtospec(massflux)
        ddzdtspec[0] -= tmpspec; ddzdtspec[1] += tmpspec
        # pressure gradient force contribution to divergence tend
        mstrm = np.empty(dz.shape, dtype=self.dtype) # montgomery streamfunction
        #mstrm[0] = self.grav*(self.orog + dz[0] + dz[1]) # with orography
        mstrm[0] = self.grav*(dz[0] + dz[1])
        mstrm[1] = mstrm[0] + (self.grav*self.delth/self.theta1)*dz[1]
        ddivdtspec += -self.ft.lap*self.ft.grdtospec(mstrm+0.5*(u**2+v**2))
        # hyperdiffusion of vorticity and divergence
        dvrtdtspec += self.hyperdiff*vrtspec
        # extra laplacian diffusion of divergence to suppress gravity waves
        ddivdtspec += self.hyperdiff*divspec + self.divlapdiff*divspec
        return dvrtdtspec,ddivdtspec,ddzdtspec

    def rk4step(self,vrtspec,divspec,dzspec):
        # update state using 4th order runge-kutta
        dt = self.dt
        k1vrt,k1div,k1thk = \
        self.gettend(vrtspec,divspec,dzspec)
        masstendspec = k1thk.sum(axis=0)
        # parameter measuring vertically integrated mass tend amplitude (external mode imbalance)
        self.masstendvar = ((masstendspec*np.conjugate(masstendspec)).real).sum() 
        k2vrt,k2div,k2thk = \
        self.gettend(vrtspec+0.5*dt*k1vrt,divspec+0.5*dt*k1div,dzspec+0.5*dt*k1thk)
        k3vrt,k3div,k3thk = \
        self.gettend(vrtspec+0.5*dt*k2vrt,divspec+0.5*dt*k2div,dzspec+0.5*dt*k2thk)
        k4vrt,k4div,k4thk = \
        self.gettend(vrtspec+dt*k3vrt,divspec+dt*k3div,dzspec+dt*k3thk)
        vrtspec += dt*(k1vrt+2.*k2vrt+2.*k3vrt+k4vrt)/6.
        divspec += dt*(k1div+2.*k2div+2.*k3div+k4div)/6.
        dzspec += dt*(k1thk+2.*k2thk+2.*k3thk+k4thk)/6.
        self.t += dt
        return vrtspec,divspec,dzspec

    def rk4step_iau(self,vrtspec,divspec,dzspec,fvrtspec,fdivspec,fdzspec):
        # update state using 4th order runge-kutta, adding extra forcing
        # that is constant over the time interval.
        dt = self.dt
        k1vrt,k1div,k1thk = \
        self.gettend(vrtspec,divspec,dzspec)
        k1vrt += fvrtspec; k1div += fdivspec; k1thk += fdzspec
        masstendspec = k1thk.sum(axis=0)
        # parameter measuring vertically integrated mass tend amplitude (external mode imbalance)
        self.masstendvar = ((masstendspec*np.conjugate(masstendspec)).real).sum() 
        k2vrt,k2div,k2thk = \
        self.gettend(vrtspec+0.5*dt*k1vrt,divspec+0.5*dt*k1div,dzspec+0.5*dt*k1thk)
        k2vrt += fvrtspec; k2div += fdivspec; k2thk += fdzspec
        k3vrt,k3div,k3thk = \
        self.gettend(vrtspec+0.5*dt*k2vrt,divspec+0.5*dt*k2div,dzspec+0.5*dt*k2thk)
        k3vrt += fvrtspec; k3div += fdivspec; k3thk += fdzspec
        k4vrt,k4div,k4thk = \
        self.gettend(vrtspec+dt*k3vrt,divspec+dt*k3div,dzspec+dt*k3thk)
        k4vrt += fvrtspec; k4div += fdivspec; k4thk += fdzspec
        vrtspec += dt*(k1vrt+2.*k2vrt+2.*k3vrt+k4vrt)/6.
        divspec += dt*(k1div+2.*k2div+2.*k3div+k4div)/6.
        dzspec += dt*(k1thk+2.*k2thk+2.*k3thk+k4thk)/6.
        self.t += dt
        return vrtspec,divspec,dzspec

    def advance(self,vrt,div,dz,grid=False):
        # advance forward number of timesteps given by 'timesteps' instance var.
        # if grid==True, inputs and outputs  are u,v,dz on grid, otherwise
        # spectral coeffs of vorticity,divergence and layer thickness
        if grid :
            vrtspec, divspec = self.ft.getvrtdivspec(vrt,div)
            dzspec = self.ft.grdtospec(dz)
        else:
            vrtspec=vrt;divspec=div;dzspec=dz
        for n in range(self.timesteps):
            vrtspec, divspec, dzspec = self.rk4step(vrtspec,divspec,dzspec)
        if grid:
            u, v = self.ft.getuv(vrtspec,divspec)
            dz = self.ft.spectogrd(dzspec)
            return u,v,dz
        else:
            return vrtspec, divspec, dzspec

# simple driver functions suitable for mulitprocessing (easy to serialize)
def run_model(u,v,dz,N,L,dt,timesteps,theta1=300,theta2=320,f=1.e-4,div2_diff_efold=1.e30,\
              zmid=5.e3,ztop=10.e3,diff_efold=6.*3600.,diff_order=8,tdrag=4*86400,tdiab=20*86400,umax=12.5,jetexp=2):
    ft = Fouriert(N,L,threads=1)
    model=TwoLayer(ft,dt,theta1=theta1,theta2=theta2,f=f,div2_diff_efold=div2_diff_efold,\
    zmid=zmid,ztop=ztop,diff_efold=diff_efold,diff_order=diff_order,tdrag=tdrag,tdiab=tdiab,umax=umax,jetexp=jetexp)
    vrtspec, divspec = ft.getvrtdivspec(u,v)
    dzspec = ft.grdtospec(dz)
    for n in range(timesteps):
        vrtspec, divspec, dzspec = model.rk4step(vrtspec,divspec,dzspec)
    u, v = ft.getuv(vrtspec,divspec)
    dz = ft.spectogrd(dzspec)
    return u,v,dz,model.masstendvar

def run_model4d(u,v,dz,N,L,dt,timesteps,theta1=300,theta2=320,f=1.e-4,div2_diff_efold=1.e30,\
                zmid=5.e3,ztop=10.e3,diff_efold=6.*3600.,diff_order=8,tdrag=4*86400,tdiab=20*86400,umax=12.5,jetexp=2):
    ft = Fouriert(N,L,threads=1)
    model=TwoLayer(ft,dt,theta1=theta1,theta2=theta2,f=f,div2_diff_efold=div2_diff_efold,\
    zmid=zmid,ztop=ztop,diff_efold=diff_efold,diff_order=diff_order,tdrag=tdrag,tdiab=tdiab,umax=umax,jetexp=jetexp)
    vrtspec, divspec = ft.getvrtdivspec(u,v)
    dzspec = ft.grdtospec(dz)
    uout = np.empty((timesteps,)+u.shape,u.dtype)
    vout = np.empty((timesteps,)+v.shape,v.dtype)
    dzout = np.empty((timesteps,)+dz.shape,dz.dtype)
    for n in range(timesteps):
        vrtspec, divspec, dzspec = model.rk4step(vrtspec,divspec,dzspec)
        u, v = ft.getuv(vrtspec,divspec)
        dz = ft.spectogrd(dzspec)
        uout[n] = u; vout[n] = v; dzout[n] = dz
    return uout,vout,dzout,model.masstendvar

def run_model_iau(u,v,dz,uinc,vinc,dzinc,wts,N,L,dt,timesteps,theta1=300,theta2=320,f=1.e-4,div2_diff_efold=1.e30,\
              zmid=5.e3,ztop=10.e3,diff_efold=6.*3600.,diff_order=8,tdrag=4*86400,tdiab=20*86400,umax=12.5,jetexp=2):
    ft = Fouriert(N,L,threads=1)
    model=TwoLayer(ft,dt,theta1=theta1,theta2=theta2,f=f,div2_diff_efold=div2_diff_efold,\
    zmid=zmid,ztop=ztop,diff_efold=diff_efold,diff_order=diff_order,tdrag=tdrag,tdiab=tdiab,umax=umax,jetexp=jetexp)
    vrtspec, divspec = ft.getvrtdivspec(u,v)
    dzspec = ft.grdtospec(dz)
    if len(uinc.shape) == 3: # 3diau
        fvrtspec, fdivspec = ft.getvrtdivspec(uinc,vinc)
        fdzspec = ft.grdtospec(dzinc)
        for n in range(timesteps):
            vrtspec, divspec, dzspec = model.rk4step_iau(vrtspec,divspec,dzspec,wts[n]*fvrtspec,wts[n]*fdivspec,wts[n]*fdzspec)
    elif len(uinc.shape) == 4: # 4diau (increments already interpolated to model timestep)
        for n in range(timesteps):
            fvrtspec, fdivspec = ft.getvrtdivspec(uinc[n],vinc[n])
            fdzspec = ft.grdtospec(dzinc[n])
            vrtspec, divspec, dzspec = model.rk4step_iau(vrtspec,divspec,dzspec,wts[n]*fvrtspec,wts[n]*fdivspec,wts[n]*fdzspec)
    else:
        raise ValueError('increments must be 3d or 4d arrays')
        
    u, v = ft.getuv(vrtspec,divspec)
    dz = ft.spectogrd(dzspec)
    return u,v,dz,model.masstendvar

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('qt5agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import os

    # grid, time step info
    N = 64  
    L = 20000.e3
    dt = 600 # time step in seconds

    # get OMP_NUM_THREADS (threads to use) from environment.
    threads = int(os.getenv('OMP_NUM_THREADS','1'))
    ft = Fouriert(N,L,threads=threads)

    # create model instance, override default parameters.
    model=TwoLayer(ft,dt)

    # vort, div initial conditions
    dtype = model.dtype
    vref = np.zeros(model.uref.shape, model.uref.dtype)
    vrtspec, divspec = model.ft.getvrtdivspec(model.uref, vref)
    vrt = model.ft.spectogrd(vrtspec)
    vrt += np.random.normal(0,2.e-6,size=(2,ft.Nt,ft.Nt)).astype(dtype)
    # add isolated blob to upper layer
    nexp = 20
    x = np.arange(0,2.*np.pi,2.*np.pi/ft.Nt); y = np.arange(0.,2.*np.pi,2.*np.pi/ft.Nt)
    x,y = np.meshgrid(x,y)
    x = x.astype(dtype); y = y.astype(dtype)
    vrt[1] = vrt[1]+2.e-6*(np.sin(x/2)**(2*nexp)*np.sin(y)**nexp)
    vrtspec = model.ft.grdtospec(vrt)
    divspec = np.zeros(vrtspec.shape, vrtspec.dtype)
    dzspec = model.nlbalance(vrtspec)
    dz = model.ft.spectogrd(dzspec)
    u,v = model.ft.getuv(vrtspec,divspec)
    vrtspec, divspec = model.ft.getvrtdivspec(u,v)
    if dz.min() < 0:
        raise ValueError('negative layer thickness! adjust jet parameters')

    # run model, animate pv
    nout = int(6.*3600./model.dt) # plot interval
    nsteps = int(200*86400./model.dt)//nout-2 # number of time steps to animate

    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(121); ax.axis('off')
    pv = (0.5*model.zmid/model.f)*(vrt + model.f)/dz
    vmin = 0; vmax = 1.75 
    plt.tight_layout()
    im1=ax.imshow(pv[0],cmap=plt.cm.jet,vmin=vmin,vmax=vmax,interpolation="nearest")
    td = model.t/86400.
    txt1=ax.text(0.5,0.95,'Lower Layer PV day %6.3f' % td,\
                ha='center',color='w',fontsize=18,transform=ax.transAxes)
    ax = fig.add_subplot(122); ax.axis('off')
    plt.tight_layout()
    im2=ax.imshow(pv[1],cmap=plt.cm.jet,vmin=vmin,vmax=vmax,interpolation="nearest")
    txt2=ax.text(0.5,0.95,'Upper Layer PV day %6.3f' % td,\
                ha='center',color='w',fontsize=18,transform=ax.transAxes)

    def updatefig(*args):
        global vrtspec, divspec, dzspec
        for n in range(nout):
            vrtspec, divspec, dzspec = model.rk4step(vrtspec, divspec,\
                    dzspec)
        vrt = model.ft.spectogrd(vrtspec)
        dz = model.ft.spectogrd(dzspec)
        pv = (0.5*model.zmid/model.f)*(vrt + model.f)/dz
        td = model.t/86400.
        im1.set_data(pv[0])
        txt1.set_text('Lower Layer PV day %7.3f' % td)
        im2.set_data(pv[1])
        txt2.set_text('Upper Layer PV day %7.3f' % td)
        return im1,txt1,im2,txt2,

    ani = animation.FuncAnimation(fig,updatefig,interval=0,frames=nsteps,repeat=False,blit=True)
    plt.show()

