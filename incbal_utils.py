# computing balanced flow (using nonlinear balance equation and unapproximated
# vorticity and continuity equations)
import numpy as np

def getincbal(model,dzb,vrtb,divb,vrt,div=None,nitermax=100,relax=0.02,eps=1.e-4,verbose=False):
    """computes balanced layer thickness and divergence increment given vorticity increment (and background)."""
 
    # first, solve non-linear balance equation to get layer thickness given vorticity
    vrtspec = model.ft.grdtospec(vrt)
    vrtspecb = model.ft.grdtospec(vrtb)
    psispec = model.ft.invlap*vrtspec
    psispecb = model.ft.invlap*vrtspecb
    psixx = model.ft.spectogrd(-ft.k**2*psispec)
    psixxb = model.ft.spectogrd(-ft.k**2*psispecb)
    psiyy = vrt - psixx
    psiyyb = vrtb - psixxb
    psixy = model.ft.spectogrd(-ft.k*ft.l*psispec)
    psixyb = model.ft.spectogrd(-ft.k*ft.l*psispecb)
    tmpspec = model.f*vrtspec + 2.*model.ft.grdtospec(psixxb*psiyy + psixx*psiyyb - 2.*psixyb*psixy)
    mspec = model.ft.invlap*tmpspec
    dzspec = np.zeros(mspec.shape, mspec.dtype)
    dzspec[0,...] = mspec[0,...]/model.theta1
    dzspec[1,...] = (mspec[1,:]-mspec[0,...])/(model.theta2-model.theta1)
    dzspec[0,...] -= dzspec[1,...]
    dzspec = (model.theta1/model.grav)*dzspec # convert from exner function to height units (m)
    dz = model.ft.spectogrd(dzspec)
    # remove area mean
    dz[0,...] = dz[0,...] - dz[0,...].mean()
    dz[1,...] = dz[1,...] - dz[1,...].mean()
    if type(div) == bool and div == False: # don't compute balanced divergence
        div = np.zeros(vrt.shape, vrt.dtype)
        return dz,div
    dzx,dzy = model.ft.getgrad(dz)
    dzspecb = model.ft.grdtospec(dzb)
    dzbx,dzby = model.ft.getgrad(dzb)
    urot = model.ft.spectogrd(-ft.il*psispec); vrot = model.ft.spectogrd(ft.ik*psispec)
    divspecb = model.ft.grdtospec(divb)
    ub, vb = model.ft.getuv(vrtspecb, divspecb)
    dvrtspecdtb, ddivspecdtb, ddzspecdtb = model.gettend(vrtspecb,divspecb,dzspecb)
    dvrtdtb = model.ft.spectogrd(dvrtspecdtb)
    massflux = (model.dzref[1] - dz[1])/model.tdiab
    massfluxb = (model.dzref[1] - dzb[1])/model.tdiab

    def nlbalance_tend(model,dvrtdt):
        # ft: Fourier transform object
        # f: coriolis param
        # grav: gravity
        # theta1,theta2: pot temp in each layer
        # dvrtdt: vorticity tendency in each layer
        # returns dz, layer thickness of each layer
        # solve tendency of nonlinear balance eqn to get layer thickness tendency
        # given vorticity tendency (psixx,psiyy and psixy already computed)
        dvrtspecdt = model.ft.grdtospec(dvrtdt)
        dpsispecdt = model.ft.invlap*dvrtspecdt
        dpsispecdtb = model.ft.invlap*dvrtspecdtb
        dpsixxdt = model.ft.spectogrd(-ft.k**2*dpsispecdt)
        dpsixxdtb = model.ft.spectogrd(-ft.k**2*dpsispecdtb)
        dpsiyydt = dvrtdt - dpsixxdt
        dpsiyydtb = dvrtdtb - dpsixxdtb
        dpsixydt = model.ft.spectogrd(-ft.k*ft.l*dpsispecdt)
        dpsixydtb = model.ft.spectogrd(-ft.k*ft.l*dpsispecdtb)
        tmpspec = model.f*dvrtspecdt + 2.*model.ft.grdtospec(dpsixxdtb*psiyy + dpsixxdt*psiyyb + 
                                                       psixxb*dpsiyydt + psixx*dpsiyydtb - 
                                                       2*psixyb*dpsixydt - 2*psixy*dpsixydtb)
        mspec = model.ft.invlap*tmpspec
        dzspec = np.zeros(mspec.shape, mspec.dtype)
        dzspec[0,...] = mspec[0,...]/model.theta1
        dzspec[1,...] = (mspec[1,:]-mspec[0,...])/(model.theta2-model.theta1)
        dzspec[0,...] -= dzspec[1,...]
        dzspec = (model.theta1/model.grav)*dzspec # convert from exner function to height units (m)
        ddzdt = model.ft.spectogrd(dzspec)
        # remove area mean
        ddzdt[0,...] = ddzdt[0,...] - ddzdt[0,...].mean()
        ddzdt[1,...] = ddzdt[1,...] - ddzdt[1,...].mean()
        return ddzdt

    # get balanced divergence computed iterative algorithm
    # following appendix of https://doi.org/10.1175/1520-0469(1993)050<1519:ACOPAB>2.0.CO;2
    # start iteration with div=0
    if div is None: # use specified initial estimate
         div = np.zeros(vrt.shape, vrt.dtype)
    converged=False
    for niter in range(nitermax):
        divspec = model.ft.grdtospec(div)
        chispec = model.ft.invlap*divspec
        udivspec = model.ft.ik*chispec; vdivspec = model.ft.il*chispec
        udiv = model.ft.spectogrd(udivspec); vdiv = model.ft.spectogrd(vdivspec)
        u = urot+udiv; v = vrot+vdiv
        # compute initial guess of vorticity tendency 
        # first, transform fields from spectral space to grid space.
        # diabatic mass flux due to interface relaxation.
        # horizontal vorticity flux
        tmp1 = ub*vrt + u*vrtb + u*model.f; tmp2 = vb*vrt + v*vrtb + v*model.f
        # add lower layer drag contribution
        tmp1[0] += v[0]/model.tdrag
        tmp2[0] += -u[0]/model.tdrag
        tmp1 += 0.5*(v[1]-v[0])*massfluxb/dzb
        tmp2 -= 0.5*(u[1]-u[0])*massfluxb/dzb
        ddivdtspec, dvrtdtspec = model.ft.getvrtdivspec(tmp1,tmp2)
        dvrtdtspec *= -1
        dvrtdtspec += model.hyperdiff*vrtspec
        dvrtdt = model.ft.spectogrd(dvrtdtspec)
        # infer layer thickness tendency from d/dt of balance eqn.
        ddzdt = nlbalance_tend(model,dvrtdt)
        # new estimate of divergence from continuity eqn (neglect diabatic mass flux term)
        tmp1[0] = massflux; tmp1[1] = -massflux
        divnew = -(1./dzb)*(ddzdt + ub*dzx + u*dzbx  + vb*dzy + v*dzby + divb*dz - tmp1)
        #divnew = -(1./dzb)*(ddzdt + ub*dzx + u*dzbx  + vb*dzy + v*dzby + divb*dz)
        divdiff = divnew-div
        div = div + relax*divdiff
        divdiffmean = np.sqrt((divdiff**2).mean())
        divmean = np.sqrt((div**2).mean())
        if verbose: print(niter, divdiffmean, divdiffmean/divmean )
        if divnew.max() > 1.e-2: break
        if divdiffmean/divmean < eps:    
            converged = True
            break
    if not converged:
        raise RuntimeError('balanced divergence solution did not converge')

    # remove area mean
    div[0]=div[0]-div[0].mean()
    div[1]=div[1]-div[1].mean()

    return dz,div

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import sys
    import numpy as np
    from pyfft import Fouriert
    from twolayer import TwoLayer

    from netCDF4 import Dataset
    filename = sys.argv[1]
    nc = Dataset(filename)

    ft = Fouriert(nc.N,nc.L,precision=str(nc['dz_b'].dtype))

    ntime = int(sys.argv[2])
    u_b = nc['u_b'][ntime]
    v_b = nc['v_b'][ntime]
    dz_b = nc['dz_b'][ntime]
    u_a = nc['u_a'][ntime]
    v_a = nc['v_a'][ntime]
    dz_a = nc['dz_a'][ntime]

    uensmean_b = u_b.mean(axis=0)
    vensmean_b = v_b.mean(axis=0)
    dzensmean_b = dz_b.mean(axis=0)
    vrtspec, divspec = model.ft.getvrtdivspec(uensmean_b,vensmean_b)
    vrtensmean_b = model.ft.spectogrd(vrtspec); divensmean_b = model.ft.spectogrd(divspec)
    uensmean_a = u_a.mean(axis=0)
    vensmean_a = v_a.mean(axis=0)
    dzensmean_a = dz_a.mean(axis=0)
    vrtspec, divspec = model.ft.getvrtdivspec(uensmean_a,vensmean_a)
    vrtensmean_a = model.ft.spectogrd(vrtspec); divensmean_a = model.ft.spectogrd(divspec)

    model = TwoLayer(ft,600.,zmid=nc.zmid,ztop=nc.ztop,tdrag=nc.tdrag,tdiab=nc.tdiab,\
    umax=nc.umax,jetexp=nc.jetexp,theta1=nc.theta1,theta2=nc.theta2,diff_efold=nc.diff_efold)

    vrtinc = vrtensmean_a-vrtensmean_b
    divinc = divensmean_a-divensmean_b
    dzinc = dzensmean_a-dzensmean_b
    dzincbal,divincbal = getincbal(model,dzensmean_b,vrtensmean_b,divensmean_b,vrtinc,div=None,\
                         nitermax=1000,relax=0.02,eps=1.e-4,verbose=False)
    dznew = dzensmean_b+dzincbal
    print('updated dz min/max',dznew.min(), dznew.max())

    #nanals = u_a.shape[0]
    #for nmem in range(nanals):
    #    uinc = u_a[nmem] - u_b[nmem]
    #    vinc = v_a[nmem] - v_b[nmem]
    #    dzinc = dz_a[nmem] - dz_b[nmem]
    #    vrtspec, divspec = model.ft.getvrtdivspec(uinc,vinc)
    #    vrtinc = model.ft.spectogrd(vrtspec); divinc = model.ft.spectogrd(divspec)
    #    vrtspec, divspec = model.ft.getvrtdivspec(u_b[nmem],v_b[nmem])
    #    vrt_b = model.ft.spectogrd(vrtspec); div_b = model.ft.spectogrd(divspec)
    #    # compute balanced layer thickness and divergence given vorticity.
    #    dzincbal,divincbal = getincbal(model,dz_b[nmem],vrt_b,div_b,vrtinc,div=None,\
    #                         nitermax=1000,relax=0.02,eps=1.e-4,verbose=False)
    #    print(nmem)
    #    print(dzinc.min(), dzinc.max())
    #    print(dzincbal.min(), dzincbal.max())
    #    print(divinc.min(), divinc.max())
    #    print(divincbal.min(), divincbal.max())
    #    dznew = dz_b[nmem]+dzincbal
    #    print('updated dz min/max',dznew.min(), dznew.max())
    #nc.close()

    nlevplot = 1
    dzincplot = dzinc[nlevplot]
    dzincbalplot = dzincbal[nlevplot]
    divincplot = divinc[nlevplot]
    divincbalplot = divincbal[nlevplot]
    dzmin=-500
    dzmax=500
    divmin=-1.e-5
    divmax=1.e-5
    divunbalmax=5.e-7
    dzunbalmax=100
    if nlevplot == 1:
        levname='upper'
    elif nlevplot == 0:
        levname='lower'
    elif nlevplot < 0:
        levname='total'
        dzplot = model.ztop - dz.sum(axis=0)
        dzbalplot = model.ztop - dzbal.sum(axis=0)
        dzmin = -200
        dzmax = 200.
        dzunbalmax = 2
        
    dzincunbalplot = dzincplot-dzincbalplot
    print(dzinc.min(), dzinc.max())
    print(dzincbal.min(), dzincbal.max())
    plt.figure()
    plt.imshow(dzincplot,cmap=plt.cm.bwr,vmin=dzmin,vmax=dzmax,interpolation="nearest")
    plt.colorbar()
    plt.title('%s layer thickness increment' % levname)
    plt.figure()
    plt.imshow(dzincbalplot,cmap=plt.cm.bwr,vmin=dzmin,vmax=dzmax,interpolation="nearest")
    plt.colorbar()
    plt.title('balanced %s layer thickness increment' % levname)

    print(divinc.min(), divinc.max())
    print(divincbal.min(), divincbal.max())
    plt.figure()
    plt.imshow(divincplot,cmap=plt.cm.bwr,vmin=divmin,vmax=divmax,interpolation="nearest")
    plt.colorbar()
    plt.title('%s layer divergence increment' % levname)
    plt.figure()
    plt.imshow(divincbalplot,cmap=plt.cm.bwr,vmin=divmin,vmax=divmax,interpolation="nearest")
    plt.colorbar()
    plt.title('balanced %s layer divergence increment' % levname)

    plt.show()
