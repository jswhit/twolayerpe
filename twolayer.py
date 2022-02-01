import numpy as np
from pyfft import Fouriert

# f-plane version two-layer baroclinic primitive equation model
# same as dry version of mc2RSW model of Lambaerts 2011 (doi:10.1063/1.3582356)

# run on command line to generate an animation.  
# needs pyfftw and matplotlib with qt5agg backend for animation.

class TwoLayer(object):

    def __init__(self,ft,dt,theta1=300,theta2=330,f=1.e-4,\
                 zmid=5.e3,ztop=10.e3,diff_efold=12*3600.,diff_order=8,tdrag=4,tdiab=20,umax=12.5,hmax=0.e3):
        # setup model parameters
        self.theta1 = np.array(theta1,np.float32) # lower layer pot. temp.
        self.theta2 = np.array(theta2,np.float32) # upper layer pot. temp.
        self.delth = np.array(theta2-theta1,np.float32) # difference in potential temp between layers
        self.hmax = np.array(hmax,np.float32) # orographic amplitude
        self.grav = 9.80616 # gravity
        self.zmid = np.array(zmid,np.float32) # resting depth of lower layer (m)
        self.ztop = np.array(ztop,np.float32) # resting depth of both layers (m)
        self.umax = np.array(umax,np.float32) # equilibrium jet strength
        self.ft = ft # Fouriert instance
        self.dt = np.array(dt,np.float32) # time step (secs)
        self.tdiab = np.array(tdiab*86400.,np.float32) # lower layer drag timescale
        self.tdrag = np.array(tdrag*86400.,np.float32) # interface relaxation timescale
        self.f = np.array(f,np.float32)
        # hyperdiffusion parameters
        self.diff_order = np.array(diff_order, np.float32)  # hyperdiffusion order
        self.diff_efold = np.array(diff_efold, np.float32)  # hyperdiff time scale
        ktot = np.sqrt(self.ft.ksqlsq)
        pi = np.array(np.pi,np.float32)  
        ktotcutoff = np.array(pi * self.ft.N / self.ft.L, np.float32)
        # integrating factor for hyperdiffusion
        self.hyperdiff = -(1./self.diff_efold)*(ktot/ktotcutoff)**self.diff_order
        # initialize orography
        x = np.arange(0, self.ft.L, self.ft.L / self.ft.Nt, dtype=np.float32)
        y = np.arange(0, self.ft.L, self.ft.L / self.ft.Nt, dtype=np.float32)
        x, y = np.meshgrid(x,y)
        self.x = x; self.y = y
        l = 2.*pi / self.ft.L
        self.orog = hmax*np.sin(l*y)*np.sin(l*x)
        # set equilibrium layer thicknes profile.
        self._interface_profile(umax)
        self.t = 0.
        self.timesteps = 1

    def _interface_profile(self,umax):
        ug = np.zeros((2,self.ft.Nt,self.ft.Nt),dtype=np.float32)
        vg = np.zeros((2,self.ft.Nt,self.ft.Nt),dtype=np.float32)
        l = np.array(2*np.pi,np.float32) / self.ft.L
        ug[1] = umax*np.sin(l*self.y)
        uspec = self.ft.grdtospec(ug)
        vrtspec, divspec = self.ft.getvrtdivspec(ug,vg)
        ug,vg = self.ft.getuv(vrtspec,divspec)
        self.uref = ug
        lyrthkspec = self.nlbalance(vrtspec)
        self.lyrthkref = self.ft.spectogrd(lyrthkspec)
        #import  matplotlib.pyplot as plt
        #print(self.lyrthkref[1].min(),self.lyrthkref[1].max())
        #plt.imshow(self.lyrthkref[1])
        #plt.colorbar()
        #plt.show()
        #mstrm = np.empty((2,self.ft.Nt,self.ft.Nt),np.float32)
        #mstrm[0] = self.grav*(self.orog + self.lyrthkref[0] + self.lyrthkref[1])
        #mstrm[1]=mstrm[0]+(self.grav*self.delth/self.theta1)*self.lyrthkref[1]
        #mx, my = self.ft.getgrad(self.ft.grdtospec(mstrm))
        #u=-my/self.f
        #print(u.min(),u.max())
        #raise SystemExit
        if self.lyrthkref.min() < 0:
            raise ValueError('negative layer thickness! adjust equilibrium jet parameter')

    def nlbalance(self,vrtspec):
        # solve nonlinear balance eqn to get layer thickness given vorticity.
        divspec = np.zeros(vrtspec.shape, vrtspec.dtype)
        lyrthkspec = np.zeros(vrtspec.shape, vrtspec.dtype)
        vrtg = self.ft.spectogrd(vrtspec)
        ug,vg = self.ft.getuv(vrtspec,divspec)
        tmpg1 = ug*(vrtg+self.f); tmpg2 = vg*(vrtg+self.f)
        tmpspec1, tmpspec2 = self.ft.getvrtdivspec(tmpg1,tmpg2)
        tmpspec2 = self.ft.grdtospec(0.5*(ug**2+vg**2))
        mspec = self.ft.invlap*tmpspec1 - tmpspec2
        mgrid = self.ft.spectogrd(mspec)
        lyrthkspec[0,...] =\
        (mspec[0,...]-self.ft.grdtospec(self.grav*self.orog))/self.theta1
        lyrthkspec[1,...] = (mspec[1,:]-mspec[0,...])/self.delth
        lyrthkspec[0,...] = lyrthkspec[0,...] - lyrthkspec[1,...]
        lyrthkspec = (self.theta1/self.grav)*lyrthkspec # convert from exner function to height units (m)
        # set area mean in grid space
        lyrthkg = self.ft.spectogrd(lyrthkspec)
        lyrthkg[0,...] = lyrthkg[0,...] - lyrthkg[0,...].mean() + self.ztop - self.zmid
        lyrthkg[1,...] = lyrthkg[1,...] - lyrthkg[1,...].mean() + self.zmid
        lyrthkspec = self.ft.grdtospec(lyrthkg)
        return lyrthkspec

    def gettend(self,vrtspec,divspec,lyrthkspec):
        # compute tendencies.
        # first, transform fields from spectral space to grid space.
        vrtg = self.ft.spectogrd(vrtspec)
        ug,vg = self.ft.getuv(vrtspec,divspec)
        lyrthkg = self.ft.spectogrd(lyrthkspec)
        # diabatic mass flux due to interface relaxation.
        massflux = (self.lyrthkref[1] - lyrthkg[1])/self.tdiab
        # horizontal vorticity flux
        tmpg1 = ug*(vrtg+self.f); tmpg2 = vg*(vrtg+self.f)
        # add lower layer drag contribution
        tmpg1[0] += vg[0]/self.tdrag
        tmpg2[0] += -ug[0]/self.tdrag
        # add diabatic momentum flux contribution
        # (this version averages vertical flux at top
        # and bottom of each layer)
        # same as 'improved' mc2RSW model (DOI: 10.1002/qj.3292)
        tmpg1 += 0.5*(vg[1]-vg[0])*massflux/lyrthkg
        tmpg2 -= 0.5*(ug[1]-ug[0])*massflux/lyrthkg
        # compute vort flux contributions to vorticity and divergence tend.
        ddivdtspec, dvrtdtspec = self.ft.getvrtdivspec(tmpg1,tmpg2)
        dvrtdtspec *= -1
        # horizontal mass flux contribution to continuity
        tmpg1 = ug*lyrthkg; tmpg2 = vg*lyrthkg
        tmpspec, dlyrthkdtspec = self.ft.getvrtdivspec(tmpg1,tmpg2)
        dlyrthkdtspec *= -1
        # diabatic mass flux contribution to continuity
        tmpspec = self.ft.grdtospec(massflux)
        dlyrthkdtspec[0] -= tmpspec; dlyrthkdtspec[1] += tmpspec
        # pressure gradient force contribution to divergence tend (includes
        # orography).
        mstrm = np.empty(lyrthkg.shape, dtype=np.float32) # montgomery streamfunction
        mstrm[0] = self.grav*(self.orog + lyrthkg[0] + lyrthkg[1])
        mstrm[1] = mstrm[0] + (self.grav*self.delth/self.theta1)*lyrthkg[1]
        ddivdtspec += -self.ft.lap*self.ft.grdtospec(mstrm+0.5*(ug**2+vg**2))
        # hyperdiffusion of vorticity and divergence
        dvrtdtspec += self.hyperdiff*vrtspec
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
    def advance(self, vrtspec, divspec, lyrthkspec):
        # advance forward number of timesteps given by 'timesteps' instance var.
        for n in range(self.timesteps):
            vrtspec, divspec, lyrthkspec = self.rk4step(vrtspec,divspec,lyrthkspec)
        return vrtspec, divspec, lyrthkspec

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
    model=TwoLayer(ft,dt,hmax=2000)

    # vort, div initial conditions
    vref = np.zeros(model.uref.shape, model.uref.dtype)
    vrtspec, divspec = model.ft.getvrtdivspec(model.uref, vref)
    vrtg = model.ft.spectogrd(vrtspec)
    vrtg += np.random.normal(0,2.e-6,size=(2,ft.Nt,ft.Nt)).astype(np.float32)
    # add isolated blob to upper layer
    nexp = 20
    x = np.arange(0,2.*np.pi,2.*np.pi/ft.Nt); y = np.arange(0.,2.*np.pi,2.*np.pi/ft.Nt)
    x,y = np.meshgrid(x,y)
    x = x.astype(np.float32); y = y.astype(np.float32)
    vrtg[1] = vrtg[1]+2.e-6*(np.sin(x/2)**(2*nexp)*np.sin(y)**nexp)
    vrtspec = model.ft.grdtospec(vrtg)
    divspec = np.zeros(vrtspec.shape, vrtspec.dtype)
    lyrthkspec = model.nlbalance(vrtspec)
    lyrthkg = model.ft.spectogrd(lyrthkspec)
    ug,vg = model.ft.getuv(vrtspec,divspec)
    vrtspec, divspec = model.ft.getvrtdivspec(ug,vg)
    if lyrthkg.min() < 0:
        raise ValueError('negative layer thickness! adjust jet parameters')

    # run model, animate pv
    nout = int(3.*3600./model.dt) # plot interval
    nsteps = int(100*86400./model.dt)//nout-2 # number of time steps to animate

    fig = plt.figure(figsize=(16,8))
    vrtspec, divspec, lyrthkspec = model.rk4step(vrtspec, divspec, lyrthkspec)
    vrtg = model.ft.spectogrd(vrtspec)
    lyrthkg = model.ft.spectogrd(lyrthkspec)
    pv = (0.5*model.zmid/model.f)*(vrtg + model.f)/lyrthkg
    vmin = 0; vmax = 2.0
    ax = fig.add_subplot(121); ax.axis('off')
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
        global vrtspec, divspec, lyrthkspec
        for n in range(nout):
            vrtspec, divspec, lyrthkspec = model.rk4step(vrtspec, divspec,\
                    lyrthkspec)
        vrtg = model.ft.spectogrd(vrtspec)
        lyrthkg = model.ft.spectogrd(lyrthkspec)
        pv = (0.5*model.zmid/model.f)*(vrtg + model.f)/lyrthkg
        td = (model.t-model.dt)/86400.
        im1.set_data(pv[0])
        txt1.set_text('Lower Layer PV day %7.3f' % td)
        im2.set_data(pv[1])
        txt2.set_text('Upper Layer PV day %7.3f' % td)
        return im1,txt1,im2,txt2,

    ani = animation.FuncAnimation(fig,updatefig,interval=0,frames=nsteps,repeat=False,blit=True)
    plt.show()

