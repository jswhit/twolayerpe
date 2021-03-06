# computing balanced flow (using nonlinear balance equation and unapproximated
# vorticity and continuity equations)
import numpy as np

def getbal(model,vrt,div=None,adiab=False,dz1mean=None,dz2mean=None,nitermax=1000,relax=0.015,eps=1.e-2,verbose=False):
    """computes balanced layer thickness and divergence given vorticity."""
 
    if dz1mean is None: 
        dz1mean = model.zmid
    if dz2mean is None:
        dz2mean = model.ztop - model.zmid
    # first, solve non-linear balance equation to get layer thickness given vorticity
    vrtspec = model.ft.grdtospec(vrt)
    psispec = model.ft.invlap*vrtspec
    psixx = model.ft.spectogrd(-model.ft.k**2*psispec)
    psiyy = vrt - psixx
    psixy = model.ft.spectogrd(-model.ft.k*model.ft.l*psispec)
    tmpspec = model.f*vrtspec + 2.*model.ft.grdtospec(psixx*psiyy - psixy**2)
    mspec = model.ft.invlap*tmpspec
    dzspec = np.zeros(mspec.shape, mspec.dtype)
    dzspec[0,...] = mspec[0,...]/model.theta1
    dzspec[1,...] = (mspec[1,:]-mspec[0,...])/(model.theta2-model.theta1)
    dzspec[0,...] -= dzspec[1,...]
    dzspec = (model.theta1/model.grav)*dzspec # convert from exner function to height units (m)
    # set area mean in grid space to state of rest value
    dz = model.ft.spectogrd(dzspec)
    dz[0,...] = dz[0,...] - dz[0,...].mean() + dz1mean
    dz[1,...] = dz[1,...] - dz[1,...].mean() + dz2mean
    if type(div) == bool and div == False: # don't compute balanced divergence
        div = np.zeros(vrt.shape, vrt.dtype)
        return dz,div
    dzx,dzy = model.ft.getgrad(dz)
    urot = model.ft.spectogrd(-model.ft.il*psispec); vrot = model.ft.spectogrd(model.ft.ik*psispec)

    def nlbalance_tend(dvrtdt):
        # solve tendency of nonlinear balance eqn to get layer thickness tendency
        # given vorticity tendency (psixx,psiyy and psixy already computed)
        dvrtspecdt = model.ft.grdtospec(dvrtdt)
        dpsispecdt = model.ft.invlap*dvrtspecdt
        dpsixxdt = model.ft.spectogrd(-model.ft.k**2*dpsispecdt)
        dpsiyydt = dvrtdt - dpsixxdt
        dpsixydt = model.ft.spectogrd(-model.ft.k*model.ft.l*dpsispecdt)
        tmpspec = model.f*dvrtspecdt + 2.*model.ft.grdtospec(dpsixxdt*psiyy + psixx*dpsiyydt - 2*psixy*dpsixydt)
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
    if div is None: # no specified initial estimate, initialize with zero
         div = np.zeros(vrt.shape, vrt.dtype)
    converged=False
    if adiab:  # use diabatic mass flux 
        massflux = np.zeros((model.ft.Nt,model.ft.Nt),model.ft.dtype)
    for niter in range(nitermax):
        divspec = model.ft.grdtospec(div)
        chispec = model.ft.invlap*divspec
        udivspec = model.ft.ik*chispec; vdivspec = model.ft.il*chispec
        udiv = model.ft.spectogrd(udivspec); vdiv = model.ft.spectogrd(vdivspec)
        u = urot+udiv; v = vrot+vdiv
        # compute initial guess of vorticity tendency 
        # first, transform fields from spectral space to grid space.
        # diabatic mass flux due to interface relaxation.
        if not adiab: massflux = (model.dzref[1] - dz[1])/model.tdiab
        # horizontal vorticity flux
        tmp1 = u*(vrt+model.f); tmp2 = v*(vrt+model.f)
        # add lower layer drag contribution
        tmp1[0] += v[0]/model.tdrag
        tmp2[0] += -u[0]/model.tdrag
        # add diabatic momentum flux contribution
        # (this version averages vertical flux at top
        # and bottom of each layer)
        # same as 'improved' mc2RSW model (DOI: 10.1002/qj.3292)
        tmp1 += 0.5*(v[1]-v[0])*massflux/dz
        tmp2 -= 0.5*(u[1]-u[0])*massflux/dz
        # compute vort flux contributions to vorticity and divergence tend.
        ddivdtspec, dvrtdtspec = model.ft.getvrtdivspec(tmp1,tmp2)
        dvrtdtspec *= -1
        dvrtdtspec += model.hyperdiff*vrtspec
        dvrtdt = model.ft.spectogrd(dvrtdtspec)
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

def pvinvert(model,pv,dzin=None,dz1mean=None,dz2mean=None,nitermax=1000,relax=0.015,eps=1.e-4,nodiv=False,adiab=False,verbose=False):
    """computes balanced layer thickness and streamfunction given potential vorticity."""

    if dz1mean is None:
        if dzin is None:
            dz1mean = model.zmid
        else:
            dz1mean = dzin[0].mean()
    if dz2mean is None:
        if dzin is None:
            dz2mean = model.ztop - model.zmid
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
        vrt = pv*(dz+dzref) - model.f
        vrtspec = model.ft.grdtospec(vrt)
        # solve nonlinear balance equation to get next estimate of dz
        psispec = model.ft.invlap*vrtspec
        psixx = model.ft.spectogrd(-model.ft.k**2*psispec)
        psiyy = vrt - psixx
        psixy = model.ft.spectogrd(-model.ft.k*model.ft.l*psispec)
        tmpspec = model.f*vrtspec + 2.*model.ft.grdtospec(psixx*psiyy - psixy**2)
        mspec = model.ft.invlap*tmpspec
        dzspec = np.zeros(mspec.shape, mspec.dtype)
        dzspec[0,...] = mspec[0,...]/model.theta1
        dzspec[1,...] = (mspec[1,:]-mspec[0,...])/(model.theta2-model.theta1)
        dzspec[0,...] -= dzspec[1,...]
        dzspec = (model.theta1/model.grav)*dzspec # convert from exner function to height units (m)
        dzprev = dz.copy()
        dz = model.ft.spectogrd(dzspec)
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
        return dz,vrt,div
    dzx,dzy = model.ft.getgrad(dz)
    urot = model.ft.spectogrd(-model.ft.il*psispec); vrot = model.ft.spectogrd(model.ft.ik*psispec)

    def nlbalance_tend(dvrtdt):
        # solve tendency of nonlinear balance eqn to get layer thickness tendency
        # given vorticity tendency (psixx,psiyy and psixy already computed)
        dvrtspecdt = model.ft.grdtospec(dvrtdt)
        dpsispecdt = model.ft.invlap*dvrtspecdt
        dpsixxdt = model.ft.spectogrd(-model.ft.k**2*dpsispecdt)
        dpsiyydt = dvrtdt - dpsixxdt
        dpsixydt = model.ft.spectogrd(-model.ft.k*model.ft.l*dpsispecdt)
        tmpspec = model.f*dvrtspecdt + 2.*model.ft.grdtospec(dpsixxdt*psiyy + psixx*dpsiyydt - 2*psixy*dpsixydt)
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
    div = np.zeros(vrt.shape, vrt.dtype)
    converged=False
    if adiab:  # use diabatic mass flux 
        massflux = np.zeros((model.ft.Nt,model.ft.Nt),model.ft.dtype)
    for niter in range(nitermax):
        divspec = model.ft.grdtospec(div)
        chispec = model.ft.invlap*divspec
        udivspec = model.ft.ik*chispec; vdivspec = model.ft.il*chispec
        udiv = model.ft.spectogrd(udivspec); vdiv = model.ft.spectogrd(vdivspec)
        u = urot+udiv; v = vrot+vdiv
        # compute initial guess of vorticity tendency 
        # first, transform fields from spectral space to grid space.
        # diabatic mass flux due to interface relaxation.
        if not adiab: massflux = (model.dzref[1] - dz[1])/model.tdiab
        # horizontal vorticity flux
        tmp1 = u*(vrt+model.f); tmp2 = v*(vrt+model.f)
        # add lower layer drag contribution
        tmp1[0] += v[0]/model.tdrag
        tmp2[0] += -u[0]/model.tdrag
        # add diabatic momentum flux contribution
        # (this version averages vertical flux at top
        # and bottom of each layer)
        # same as 'improved' mc2RSW model (DOI: 10.1002/qj.3292)
        tmp1 += 0.5*(v[1]-v[0])*massflux/dz
        tmp2 -= 0.5*(u[1]-u[0])*massflux/dz
        # compute vort flux contributions to vorticity and divergence tend.
        ddivdtspec, dvrtdtspec = model.ft.getvrtdivspec(tmp1,tmp2)
        dvrtdtspec *= -1
        dvrtdtspec += model.hyperdiff*vrtspec
        dvrtdt = model.ft.spectogrd(dvrtdtspec)
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

    return dz,vrt,div

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('qt5agg')
    import matplotlib.pyplot as plt
    import sys
    import numpy as np
    from pyfft import Fouriert
    from twolayer import TwoLayer

    from netCDF4 import Dataset
    filename = sys.argv[1]
    nc = Dataset(filename)

    ft = Fouriert(nc.N,nc.L,precision=str(nc['dz'].dtype))
    model = TwoLayer(ft,nc.dt,zmid=nc.zmid,ztop=nc.ztop,tdrag=nc.tdrag,tdiab=nc.tdiab,\
    umax=nc.umax,jetexp=nc.jetexp,theta1=nc.theta1,theta2=nc.theta2,diff_efold=nc.diff_efold)

    ntime = int(sys.argv[2])

    u = nc['u'][ntime]
    v = nc['v'][ntime]
    dz = nc['dz'][ntime]
    print('area mean layer thickness = ',dz[0].mean(), dz[1].mean())
    vrtspec, divspec = model.ft.getvrtdivspec(u,v)
    vrt = model.ft.spectogrd(vrtspec); div = model.ft.spectogrd(divspec)
    pv = (vrt + model.f)/dz
    # invert pv
    dzbal, vrtbal, divbal = pvinvert(model,pv,dzin=dz,\
                                     relax=0.015,eps=1.e-4,verbose=True)
    print('after pv inversion:')
    print(dz.min(), dz.max())
    print(dzbal.min(), dzbal.max())
    print((dz-dzbal).min(), (dz-dzbal).max())
    print(vrt.min(), vrt.max())
    print(vrtbal.min(), vrtbal.max())
    print(div.min(), div.max())
    print(divbal.min(), divbal.max())

    nlevplot = 1
    dzplot = dz[nlevplot]
    dzbalplot = dzbal[nlevplot]
    vrtplot = vrt[nlevplot]
    vrtbalplot = vrtbal[nlevplot]
    divplot = div[nlevplot]
    divbalplot = divbal[nlevplot]
    dzmin=0
    dzmax=1.e4
    vrtmin=-1.e-4
    vrtmax=1.e-4
    divmin=-1.e-5
    divmax=1.e-5
    vrtunbalmax=1.e-6
    divunbalmax=5.e-7
    dzunbalmax=40
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

    dzunbalplot = dzplot-dzbalplot
    print(dzunbalplot.min(), dzunbalplot.max())
    plt.figure()
    plt.imshow(dzplot,cmap=plt.cm.bwr,vmin=dzmin,vmax=dzmax,interpolation="nearest")
    plt.colorbar()
    plt.title('%s layer thickness' % levname)
    plt.figure()
    plt.imshow(dzbalplot,cmap=plt.cm.bwr,vmin=dzmin,vmax=dzmax,interpolation="nearest")
    plt.colorbar()
    plt.title('balanced %s layer thickness from pv' % levname)
    plt.figure()
    plt.imshow(dzunbalplot,cmap=plt.cm.bwr,vmin=-dzunbalmax,vmax=dzunbalmax,interpolation="nearest")
    plt.colorbar()
    plt.title('unbalanced %s layer thickness from pv' % levname)

    vrtunbalplot = vrtplot-vrtbalplot
    print('vrt min/max',vrtplot.min(), vrtplot.max())
    print('vrtbal min/max',vrtbalplot.min(), vrtbalplot.max())
    print(vrtunbalplot.min(), vrtunbalplot.max())
    plt.figure()
    plt.imshow(vrtplot,cmap=plt.cm.bwr,vmin=vrtmin,vmax=vrtmax,interpolation="nearest")
    plt.colorbar()
    plt.title('%s layer vorticity' % levname)
    plt.figure()
    plt.imshow(vrtbalplot,cmap=plt.cm.bwr,vmin=vrtmin,vmax=vrtmax,interpolation="nearest")
    plt.colorbar()
    plt.title('balanced %s layer vorticity from pv' % levname)
    plt.figure()
    plt.imshow(vrtunbalplot,cmap=plt.cm.bwr,vmin=-vrtunbalmax,vmax=vrtunbalmax,interpolation="nearest")
    plt.colorbar()
    plt.title('unbalanced %s layer vorticity from pv' % levname)

    divunbalplot = divplot-divbalplot
    print('div min/max',divplot.min(), divplot.max())
    print('divbal min/max',divbalplot.min(), divbalplot.max())
    print(divunbalplot.min(), divunbalplot.max())
    plt.figure()
    plt.imshow(divplot,cmap=plt.cm.bwr,vmin=divmin,vmax=divmax,interpolation="nearest")
    plt.colorbar()
    plt.title('%s layer divergence' % levname)
    plt.figure()
    plt.imshow(divbalplot,cmap=plt.cm.bwr,vmin=divmin,vmax=divmax,interpolation="nearest")
    plt.colorbar()
    plt.title('balanced %s layer divergence from pv' % levname)
    plt.figure()
    plt.imshow(divunbalplot,cmap=plt.cm.bwr,vmin=-divunbalmax,vmax=divunbalmax,interpolation="nearest")
    plt.colorbar()
    plt.title('unbalanced %s layer divergence from pv' % levname)

    # compute balanced layer thickness and divergence given vorticity.
    dzbal,divbal = getbal(model,vrt,div=None,adiab=True,dz1mean=dz[0].mean(),dz2mean=dz[1].mean(),\
                   nitermax=500,relax=0.02,eps=1.e-4,verbose=True)

    divbalplot = divbal[nlevplot]
    dzbalplot = dzbal[nlevplot]
    dzunbalplot = dzplot-dzbalplot
    print(dzunbalplot.min(), dzunbalplot.max())
    #plt.figure()
    #plt.imshow(dzplot,cmap=plt.cm.bwr,vmin=dzmin,vmax=dzmax,interpolation="nearest")
    #plt.colorbar()
    #plt.title('%s layer thickness' % levname)
    plt.figure()
    plt.imshow(dzbalplot,cmap=plt.cm.bwr,vmin=dzmin,vmax=dzmax,interpolation="nearest")
    plt.colorbar()
    plt.title('balanced %s layer thickness from vort' % levname)
    plt.figure()
    plt.imshow(dzunbalplot,cmap=plt.cm.bwr,vmin=-dzunbalmax,vmax=dzunbalmax,interpolation="nearest")
    plt.colorbar()
    plt.title('unbalanced %s layer thickness from vort' % levname)

    divunbalplot = divplot-divbalplot
    print('div min/max',divplot.min(), divplot.max())
    print('divbal min/max',divbalplot.min(), divbalplot.max())
    print(divunbalplot.min(), divunbalplot.max())
    #plt.figure()
    #plt.imshow(divplot,cmap=plt.cm.bwr,vmin=divmin,vmax=divmax,interpolation="nearest")
    #plt.colorbar()
    #plt.title('%s layer divergence' % levname)
    plt.figure()
    plt.imshow(divbalplot,cmap=plt.cm.bwr,vmin=divmin,vmax=divmax,interpolation="nearest")
    plt.colorbar()
    plt.title('balanced %s layer divergence from vort' % levname)
    plt.figure()
    plt.imshow(divunbalplot,cmap=plt.cm.bwr,vmin=-divunbalmax,vmax=divunbalmax,interpolation="nearest")
    plt.colorbar()
    plt.title('unbalanced %s layer divergence from vort' % levname)

    plt.show()
