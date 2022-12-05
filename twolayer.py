import numpy as np
from pyfft import Fouriert

# f-plane version two-layer baroclinic primitive equation model
# same as dry version of mc2RSW model of Lambaerts 2011 (doi:10.1063/1.3582356)

# run on command line to generate an animation.  
# needs pyfftw and matplotlib with qt5agg backend for animation.

class TwoLayer(object):

    def __init__(self,ft,dt,theta1=300,theta2=320,f=1.e-4,\
                 zmid=5.e3,ztop=10.e3,diff_efold=6*3600.,diff_order=8,\
                 div2_diff_efold=1.e30,tdrag1=10*86400,tdrag2=10.*86400,tdiab=15*86400,umax=8):
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
        self.ft = ft # Fouriert instance (defines domain size, grid resolution)
        self.dt = np.array(dt,dtype) # time step (secs)
        self.tdiab = np.array(tdiab,dtype) # interface relaxation time scalee (secs)
        self.tdrag = np.array([tdrag1,tdrag2],dtype) # linear drag time scale in each layer (secs)
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
        # set equilibrium layer thicknes profile (cos(2*pi*y/L structure).
        l = np.array(2*np.pi,self.dtype) / self.ft.L
        self.dzref = (self.f*self.theta1*umax*np.cos(l*self.y)/(l*self.grav*self.delth))+self.zmid
        if self.dzref.min() < 0:
            raise ValueError('negative layer thickness! adjust model parameters')
        self.t = 0.
        self.timesteps = 1
        self.masstendvar = 0

    def gettend(self,vrtspec,divspec,dzspec,masstend_diag=False):
        # compute tendencies.
        # first, transform fields from spectral space to grid space.
        vrt = self.ft.spectogrd(vrtspec)
        u,v = self.ft.getuv(vrtspec,divspec)
        dz = self.ft.spectogrd(dzspec)
        # diabatic mass flux due to interface relaxation.
        massflux = (self.dzref - dz[1])/self.tdiab
        # horizontal vorticity flux
        tmpg1 = u*(vrt+self.f); tmpg2 = v*(vrt+self.f)
        # add linear drag contribution
        tmpg1 += v/self.tdrag[:,np.newaxis,np.newaxis]
        tmpg2 += -u/self.tdrag[:,np.newaxis,np.newaxis]
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
        if masstend_diag:
            # mean abs total mass tend (meters/hour)
            self.masstendvar = 3600.*np.abs((self.ft.spectogrd(ddzdtspec)).sum(axis=0)).mean()
            #print(self.t,self.masstendvar)
        ddzdtspec *= -1
        # diabatic mass flux contribution to continuity
        tmpspec = self.ft.grdtospec(massflux)
        ddzdtspec[0] -= tmpspec; ddzdtspec[1] += tmpspec
        # pressure gradient force contribution to divergence tend
        mstrm = np.empty(dz.shape, dtype=self.dtype) # montgomery streamfunction
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
        self.gettend(vrtspec,divspec,dzspec,masstend_diag=True)
        #masstendspec = k1thk.sum(axis=0)
        # parameter measuring vertically integrated mass tend amplitude (external mode imbalance)
        #self.masstendvar = ((masstendspec*np.conjugate(masstendspec)).real).sum() 
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

    def nlbalance(self,vrtspec):
        """computes balanced layer thickness given vorticity (from nonlinear bal eqn)"""
        dzspec = np.zeros(vrtspec.shape, vrtspec.dtype)
        psispec = self.ft.invlap*vrtspec
        psixx = self.ft.spectogrd(-self.ft.k**2*psispec)
        psiyy = self.ft.spectogrd(-self.ft.l**2*psispec)
        psixy = self.ft.spectogrd(self.ft.k*self.ft.l*psispec)
        tmpspec = self.f*vrtspec + 2.*self.ft.grdtospec(psixx*psiyy - psixy**2)
        mspec = self.ft.invlap*tmpspec
        # alternate form of NBE
        #divspec = np.zeros(vrtspec.shape, vrtspec.dtype)
        #vrt = self.ft.spectogrd(vrtspec)
        #u,v = self.ft.getuv(vrtspec,divspec)
        #tmpg1 = u*(vrt+self.f); tmpg2 = v*(vrt+self.f)
        #tmpspec1, tmpspec2 = self.ft.getvrtdivspec(tmpg1,tmpg2)
        #tmpspec2 = self.ft.grdtospec(0.5*(u**2+v**2))
        #mspec = self.ft.invlap*tmpspec1 - tmpspec2
        dzspec[0,...] = mspec[0,...]/self.theta1
        dzspec[1,...] = (mspec[1,:]-mspec[0,...])/self.delth
        dzspec[0,...] = dzspec[0,...] - dzspec[1,...]
        dzspec = (self.theta1/self.grav)*dzspec # convert from exner function to height units (m)
        # set area mean in grid space to state of rest value
        dz = self.ft.spectogrd(dzspec)
        dz[0,...] = dz[0,...] - dz[0,...].mean() + self.zmid
        dz[1,...] = dz[1,...] - dz[1,...].mean() + self.ztop - self.zmid
        return dz

# simple driver function suitable for mulitprocessing (easy to serialize)
def run_model(u,v,dz,N,L,dt,timesteps,theta1=300,theta2=320,f=1.e-4,div2_diff_efold=1.e30,\
              zmid=5.e3,ztop=10.e3,diff_efold=6.*3600.,diff_order=8,tdrag1=10*86400,tdrag2=10.*86400,tdiab=15*86400,umax=8):
    ft = Fouriert(N,L,threads=1)
    model=TwoLayer(ft,dt,theta1=theta1,theta2=theta2,f=f,div2_diff_efold=div2_diff_efold,\
    zmid=zmid,ztop=ztop,diff_efold=diff_efold,diff_order=diff_order,tdrag1=tdrag1,tdrag2=tdrag2,tdiab=tdiab,umax=umax)
    vrtspec, divspec = ft.getvrtdivspec(u,v)
    dzspec = ft.grdtospec(dz)
    for n in range(timesteps):
        vrtspec, divspec, dzspec = model.rk4step(vrtspec,divspec,dzspec)
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

    # create model instance.
    # symmetric jet (linear drag in both layers, eddy stats same in both layers)
    model=TwoLayer(ft,dt,theta1=300,theta2=320,umax=8,tdrag1=10*86400,tdrag2=10.*86400,tdiab=15*86400)
    # asymmetric jet (linear drag only in lower layer)
    #model=TwoLayer(ft,dt,theta1=300,theta2=320,umax=8,tdrag1=5*86400,tdrag2=1.e30,tdiab=15*86400)

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
    vrt = model.ft.spectogrd(vrtspec)
    ampvrt = 2.e-6
    vrt += np.random.normal(0,ampvrt,size=(2,ft.Nt,ft.Nt)).astype(dtype)
    # add isolated blob to upper layer
    nexp = 20
    x = np.arange(0,2.*np.pi,2.*np.pi/ft.Nt); y = np.arange(0.,2.*np.pi,2.*np.pi/ft.Nt)
    x,y = np.meshgrid(x,y)
    x = x.astype(dtype); y = y.astype(dtype)
    vrt[1] = vrt[1]+ampvrt*(np.sin(x/2)**(2*nexp)*np.sin(y)**nexp)
    vrtspec = model.ft.grdtospec(vrt)
    divspec = np.zeros(vrtspec.shape, vrtspec.dtype)
    dz = model.nlbalance(vrtspec)
    dzspec = model.ft.grdtospec(dz)
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
            vrtspec, divspec, dzspec = model.rk4step(vrtspec, divspec, dzspec)
        vrt = model.ft.spectogrd(vrtspec)
        dz = model.ft.spectogrd(dzspec)
        #print(dz[0].mean(), dz[1].mean())
        pv = (0.5*model.zmid/model.f)*(vrt + model.f)/dz
        td = model.t/86400.
        im1.set_data(pv[0])
        txt1.set_text('Lower Layer PV day %7.3f' % td)
        im2.set_data(pv[1])
        txt2.set_text('Upper Layer PV day %7.3f' % td)
        return im1,txt1,im2,txt2,

    ani = animation.FuncAnimation(fig,updatefig,interval=0,frames=nsteps,repeat=False,blit=True)
    plt.show()

