import numpy as np
from pyfft import Fouriert
from twolayer import TwoLayer as TwoLayer_base

class TwoLayer(TwoLayer_base):

    def rk4step_iau(self,vrtspec,divspec,dzspec,fvrtspec,fdivspec,fdzspec):
        # update state using 4th order runge-kutta, adding extra forcing
        # that is constant over the time interval.
        dt = self.dt
        k1vrt,k1div,k1thk = \
        self.gettend(vrtspec,divspec,dzspec,masstend_diag=True)
        k1vrt += fvrtspec; k1div += fdivspec; k1thk += fdzspec
        #masstendspec = k1thk.sum(axis=0)
        # parameter measuring vertically integrated mass tend amplitude (external mode imbalance)
        #self.masstendvar = ((masstendspec*np.conjugate(masstendspec)).real).sum() 
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

    def _nlbalance_tend(self,dvrtdt,psixx,psiyy,psixy,linbal=False):
        # solve tendency of nonlinear balance eqn to get layer thickness tendency
        # given vorticity tendency (psixx,psiyy,psixy already computed)
        dvrtspecdt = self.ft.grdtospec(dvrtdt)
        dpsispecdt = self.ft.invlap*dvrtspecdt
        if linbal:
            mspec = self.f*dpsispecdt
        else:
            dpsixxdt = self.ft.spectogrd(-self.ft.k**2*dpsispecdt)
            dpsiyydt = dvrtdt - dpsixxdt
            dpsixydt = self.ft.spectogrd(-self.ft.k*self.ft.l*dpsispecdt)
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

    def _nlbal_div(self,vrtspec,dz,div,linbal=False,\
                   nitermax=1000, relax=0.015, eps=1.e-4, verbose=False):
        # iteratively solve for balanced divergence.
        dzx,dzy = self.ft.getgrad(dz)
        psispec = self.ft.invlap*vrtspec
        psixx = self.ft.spectogrd(-self.ft.k**2*psispec)
        psiyy = self.ft.spectogrd(-self.ft.l**2*psispec)
        psixy = self.ft.spectogrd(-self.ft.k*self.ft.l*psispec)
        urot = self.ft.spectogrd(-self.ft.il*psispec); vrot = self.ft.spectogrd(self.ft.ik*psispec)
        vrt = self.ft.spectogrd(vrtspec)

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
            # add linear layer drag contribution
            tmp1 += v/self.tdrag[:,np.newaxis,np.newaxis]
            tmp2 += -u/self.tdrag[:,np.newaxis,np.newaxis]
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
            ddzdt = self._nlbalance_tend(dvrtdt,psixx,psiyy,psixy,linbal=linbal)
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
        else:
            # remove area mean div 
            div[0]=div[0]-div[0].mean()
            div[1]=div[1]-div[1].mean()
            return div

    def _nlbalinc_tend(self,dvrtspecdt):
        # solve tendency of linear balance eqn to get layer thickness tendency
        dpsispecdt = self.ft.invlap*dvrtspecdt
        mspec = self.f*dpsispecdt
        dzspec = np.zeros(mspec.shape, mspec.dtype)
        dzspec[0,...] = mspec[0,...]/self.theta1
        dzspec[1,...] = (mspec[1,:]-mspec[0,...])/(self.theta2-self.theta1)
        dzspec[0,...] -= dzspec[1,...]
        dzspec = (self.theta1/self.grav)*dzspec # convert from exner function to height units (m)
        ddzdt = self.ft.spectogrd(dzspec)
        return ddzdt

    def _nlbalinc_div(self,vrtspecb,divspecb,dzb,vrtspec,dz,div=None,\
                   nitermax=1000, relax=0.015, eps=1.e-4, verbose=False):
        # iteratively solve for balanced divergence.
        dzx,dzy = self.ft.getgrad(dz)
        psispec = self.ft.invlap*vrtspec
        urot = self.ft.spectogrd(-self.ft.il*psispec); vrot = self.ft.spectogrd(self.ft.ik*psispec)
        vrt = self.ft.spectogrd(vrtspec)
        psispecb = self.ft.invlap*vrtspecb

# compute ub,vb,dzbx,dzby,divb
        divspec_zero=np.zeros_like(divspecb)
        ub, vb = self.ft.getuv(vrtspecb,divspecb)
        dzbx,dzby = self.ft.getgrad(dzb)
        divb = self.ft.spectogrd(divspecb)
        vrtb = self.ft.spectogrd(vrtspecb)
        if div is None:
            div = np.zeros_like(divb)

        # get balanced divergence computed iterative algorithm
        # following appendix of https://doi.org/10.1175/1520-0469(1993)050<1519:ACOPAB>2.0.CO;2
        # start iteration with div=0
        converged=False

        # use full (unapproximated) vort and thickness equations
        for niter in range(nitermax):
            divspec = self.ft.grdtospec(div)
            chispec = self.ft.invlap*divspec
            udivspec = self.ft.ik*chispec; vdivspec = self.ft.il*chispec
            udiv = self.ft.spectogrd(udivspec); vdiv = self.ft.spectogrd(vdivspec)
            u = urot+udiv; v = vrot+vdiv
            massflux = -dz[1]/self.tdiab
            # horizontal vorticity flux
            tmp1 = u*(vrtb+self.f) + ub*vrt
            tmp2 = v*(vrtb+self.f) + vb*vrt
            # add linearr layer drag contribution
            tmp1 += v/self.tdrag[:,np.newaxis,np.newaxis]
            tmp2 += -u/self.tdrag[:,np.newaxis,np.newaxis]
            # compute vort flux contributions to vorticity and divergence tend.
            ddivdtspec, dvrtdtspec = self.ft.getvrtdivspec(tmp1,tmp2)
            dvrtdtspec *= -1
            dvrtdtspec += self.hyperdiff*vrtspec
            # infer layer thickness tendency from d/dt of linear balance eqn.
            ddzdt = self._nlbalinc_tend(dvrtdtspec)
            # new estimate of divergence from continuity eqn
            tmp1[0] = massflux; tmp1[1] = -massflux
            divnew = -(1./dzb)*(ddzdt + ub*dzx + u*dzbx + vb*dzy + v*dzby + dz*divb + dzb*div- tmp1)
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
        else:
            # remove area mean div 
            div[0]=div[0]-div[0].mean()
            div[1]=div[1]-div[1].mean()
            return div

    def nlbalinc(self,vrtspecb,divspecb,dzb,vrtspec,div=None,linbal=False,baldiv=False,\
                 nitermax=1000, relax=0.02, eps=1.e-3, verbose=False):
        """computes linearized incremental balanced layer thickness given vorticity (from nonlinear bal eqn)"""
        dzspec = np.zeros(vrtspec.shape, vrtspec.dtype)
        psispec = self.ft.invlap*vrtspec
        if linbal:
            mspec = self.f*psispec
        else:
            psispecb = self.ft.invlap*vrtspecb
            psixx = self.ft.spectogrd(-self.ft.k**2*psispec)
            psixxb = self.ft.spectogrd(-self.ft.k**2*psispecb)
            psiyy = self.ft.spectogrd(-self.ft.l**2*psispec)
            psiyyb = self.ft.spectogrd(-self.ft.l**2*psispecb)
            psixy = self.ft.spectogrd(self.ft.k*self.ft.l*psispec)
            psixyb = self.ft.spectogrd(self.ft.k*self.ft.l*psispecb)
            tmpspec = self.f*vrtspec + 2.*self.ft.grdtospec(psixxb*psiyy + psixx*psiyyb - 2.*psixyb*psixy)
            mspec = self.ft.invlap*tmpspec
        dzspec[0,...] = mspec[0,...]/self.theta1
        dzspec[1,...] = (mspec[1,:]-mspec[0,...])/self.delth
        dzspec[0,...] = dzspec[0,...] - dzspec[1,...]
        dzspec = (self.theta1/self.grav)*dzspec # convert from exner function to height units (m)
        # set area mean in grid space to zero
        dz = self.ft.spectogrd(dzspec)
        dz[0,...] = dz[0,...] - dz[0,...].mean()
        dz[1,...] = dz[1,...] - dz[1,...].mean()
        if baldiv:
            div = self._nlbalinc_div(vrtspecb,divspecb,dzb,vrtspec,dz,div=div,\
                                     nitermax=nitermax, relax=relax, eps=eps, verbose=verbose)
            return dz,div
        else:
            return dz,np.zeros_like(dz)

    def nlbalance(self,vrtspec,div=False, dz1mean=None, dz2mean=None,\
                  nitermax=1000, relax=0.015, eps=1.e-4, verbose=False, linbal=False):
        """computes balanced layer thickness given vorticity (from nonlinear bal eqn)"""
        if dz1mean is None: 
            dz1mean = self.zmid
        if dz2mean is None:
            dz2mean = self.ztop - self.zmid
        dzspec = np.zeros(vrtspec.shape, vrtspec.dtype)

        psispec = self.ft.invlap*vrtspec

        if linbal:
            mspec = self.f*psispec
        else:
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
        #(mspec[0,...]-self.ft.grdtospec(self.grav*self.orog))/self.theta1 # with orography
        dzspec[1,...] = (mspec[1,:]-mspec[0,...])/self.delth
        dzspec[0,...] = dzspec[0,...] - dzspec[1,...]
        dzspec = (self.theta1/self.grav)*dzspec # convert from exner function to height units (m)

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

        # get balanced divergence computed iterative algorithm
        # following appendix of https://doi.org/10.1175/1520-0469(1993)050<1519:ACOPAB>2.0.CO;2
        div = self._nlbal_div(vrtspec,dz,div,linbal=linbal,\
                              nitermax=nitermax, relax=relax, eps=eps, verbose=verbose)

        return dz,div

    def pvinvert(self,pv,dzin=None,dz1mean=None,dz2mean=None,\
                 nitermax=1000,relax=0.015,eps=1.e-4,nodiv=False,verbose=False):
        """computes balanced layer thickness and streamfunction given potential vorticity."""
        if dz1mean is None:
            if dzin is None:
                dz1mean = self.zmid
            else:
                dz1mean = dzin[0].mean()
        if dz2mean is None:
            if dzin is None:
                dz2mean = self.ztop - self.zmid
            else:
                dz2mean = dzin[1].mean()
        if dzin is None:
            dz = np.zeros(pv.shape, pv.dtype)
            dz[0] = dz1mean; dz[1] = dz2mean
        else:
            dz = dzin.copy()
        dzref = np.empty(dz.shape,dz.dtype); dzref[0] = dz1mean; dzref[1] = dz2mean
        dz -= dzref
        converged = False
        for niter in range(nitermax):
            # compute vorticity from PV using initial guess of dz
            vrt = pv*(dz+dzref) - self.f
            vrtspec = self.ft.grdtospec(vrt)
            dzprev = dz.copy()
            dz,div = self.nlbalance(vrtspec,div=False, dz1mean=0., dz2mean=0.)
            # solve nonlinear balance equation to get next estimate of dz
            #psispec = self.ft.invlap*vrtspec
            #psixx = self.ft.spectogrd(-self.ft.k**2*psispec)
            #psiyy = vrt - psixx
            #psixy = self.ft.spectogrd(-self.ft.k*self.ft.l*psispec)
            #tmpspec = self.f*vrtspec + 2.*self.ft.grdtospec(psixx*psiyy - psixy**2)
            #mspec = self.ft.invlap*tmpspec
            #dzspec = np.zeros(mspec.shape, mspec.dtype)
            #dzspec[0,...] = mspec[0,...]/self.theta1
            #dzspec[1,...] = (mspec[1,:]-mspec[0,...])/(self.theta2-self.theta1)
            #dzspec[0,...] -= dzspec[1,...]
            #dzspec = (self.theta1/self.grav)*dzspec # convert from exner function to height units (m)
            #dzprev = dz.copy()
            #dz = self.ft.spectogrd(dzspec)
            dzdiff = dz-dzprev
            dz = dzprev + relax*dzdiff
            dzdiffmean = np.sqrt((dzdiff**2).mean())
            dzmean = np.sqrt((dz**2).mean())
            if verbose: print(niter, dzdiffmean, dzdiffmean/dzmean)
            if dzdiffmean/dzmean < eps:
                converged = True
                break
        if not converged:
            raise RuntimeError('pv inversion solution did not converge')
    
        dz += dzref
        if nodiv: # don't compute balanced divergence
            div = np.zeros(vrt.shape, vrt.dtype)
        else:
            # get balanced divergence computed iterative algorithm
            # following appendix of https://doi.org/10.1175/1520-0469(1993)050<1519:ACOPAB>2.0.CO;2
            div = self._nlbal_div(vrtspec,dz,div,\
                                  nitermax=nitermax, relax=relax, eps=eps, verbose=verbose)
        return dz,vrt,div

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

def run_model4d(u,v,dz,N,L,dt,timesteps,theta1=300,theta2=320,f=1.e-4,div2_diff_efold=1.e30,\
                zmid=5.e3,ztop=10.e3,diff_efold=6.*3600.,diff_order=8,tdrag1=10*86400,tdrag2=10*86400,tdiab=20*86400,umax=8):
    ft = Fouriert(N,L,threads=1)
    model=TwoLayer(ft,dt,theta1=theta1,theta2=theta2,f=f,div2_diff_efold=div2_diff_efold,\
    zmid=zmid,ztop=ztop,diff_efold=diff_efold,diff_order=diff_order,tdrag1=tdrag1,tdrag2=tdrag2,tdiab=tdiab,umax=umax)
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
              zmid=5.e3,ztop=10.e3,diff_efold=6.*3600.,diff_order=8,tdrag1=10*86400,tdrag2=10*86400,tdiab=20*86400,umax=8):
    ft = Fouriert(N,L,threads=1)
    model=TwoLayer(ft,dt,theta1=theta1,theta2=theta2,f=f,div2_diff_efold=div2_diff_efold,\
    zmid=zmid,ztop=ztop,diff_efold=diff_efold,diff_order=diff_order,tdrag1=tdrag1,tdrag2=tdrag2,tdiab=tdiab,umax=umax)
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
