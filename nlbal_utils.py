def nlbalance(ft,vrt,f=1.e-4,theta1=300,theta2=330,grav=9.8066,dz1mean=5.e3,dz2mean=5.e3):
    # ft: Fourier transform object
    # f: coriolis param
    # grav: gravity
    # theta1,theta2: pot temp in each layer
    # dz1mean, dz2mean: area mean layer thicknesses (state of rest)
    # vrt: vorticity in each layer
    # returns dz, layer thickness of each layer
    # solve nonlinear balance eqn to get layer thickness given vorticity
    vrtspec = ft.grdtospec(vrt)
    psispec = ft.invlap*vrtspec
    psixx = ft.spectogrd(-ft.k**2*psispec)
    #psiyy = ft.spectogrd(-ft.l**2*psispec)
    psiyy = vrt - psixx
    psixy = ft.spectogrd(-ft.k*ft.l*psispec)
    tmpspec = f*vrtspec + 2.*ft.grdtospec(psixx*psiyy - psixy**2)
    mspec = ft.invlap*tmpspec
    dzspec = np.zeros(mspec.shape, mspec.dtype)
    dzspec[0,...] = mspec[0,...]/theta1
    dzspec[1,...] = (mspec[1,:]-mspec[0,...])/(theta2-theta1)
    dzspec[0,...] -= dzspec[1,...]
    dzspec = (theta1/grav)*dzspec # convert from exner function to height units (m)
    # set area mean in grid space to state of rest value
    dz = ft.spectogrd(dzspec)
    dz[0,...] = dz[0,...] - dz[0,...].mean() + dz1mean
    dz[1,...] = dz[1,...] - dz[1,...].mean() + dz2mean
    return dz

def nlbalance2(ft,vrt,f=1.e-4,theta1=300,theta2=330,grav=9.8066,dz1mean=5.e3,dz2mean=5.e3):
    # solve nonlinear balance eqn to get layer thickness given vorticity.
    vrtspec = ft.grdtospec(vrt)
    divspec = np.zeros(vrtspec.shape, vrtspec.dtype)
    dzspec = np.zeros(vrtspec.shape, vrtspec.dtype)
    u,v = ft.getuv(vrtspec,divspec)
    tmp1 = u*(vrt+f); tmp2 = v*(vrt+f)
    tmpspec1, tmpspec2 = ft.getvrtdivspec(tmp1,tmp2)
    tmpspec2 = ft.grdtospec(0.5*(u**2+v**2))
    mspec = ft.invlap*tmpspec1 - tmpspec2
    dzspec[0,...] = mspec[0,...]/theta1
    dzspec[1,...] = (mspec[1,:]-mspec[0,...])/(theta2-theta1)
    dzspec[0,...] -=  dzspec[1,...]
    dzspec = (theta1/grav)*dzspec # convert from exner function to height units (m)
    # set area mean in grid space to state of rest value
    dz = ft.spectogrd(dzspec)
    dz = ft.spectogrd(dzspec)
    dz[0,...] = dz[0,...] - dz[0,...].mean() + dz1mean
    dz[1,...] = dz[1,...] - dz[1,...].mean() + dz2mean
    return dz

def nlbalance_tend(ft,vrt,dvrtdt,f=1.e-4,theta1=300,theta2=330,grav=9.8066):
    # ft: Fourier transform object
    # f: coriolis param
    # grav: gravity
    # theta1,theta2: pot temp in each layer
    # vrt: vorticity in each layer
    # dvrtdt: vorticity tendency in each layer
    # returns dz, layer thickness of each layer
    # solve nonlinear balance eqn to get layer thickness given vorticity
    dvrtspecdt = ft.grdtospec(dvrtdt)
    dpsispecdt = ft.invlap*dvrtspecdt
    vrtspec = ft.grdtospec(vrt)
    psispec = ft.invlap*vrtspec
    psixx = ft.spectogrd(-ft.k**2*psispec)
    #psiyy = ft.spectogrd(-ft.l**2*psispec)
    psiyy = vrt - psixx
    dpsixxdt = ft.spectogrd(-ft.k**2*dpsispecdt)
    #dpsiyydt = ft.spectogrd(-ft.l**2*dpsispecdt)
    dpsiyydt = dvrtdt - dpsixxdt
    psixy = ft.spectogrd(-ft.k*ft.l*psispec)
    dpsixydt = ft.spectogrd(-ft.k*ft.l*dpsispecdt)
    tmpspec = f*dvrtspecdt + 2.*ft.grdtospec(dpsixxdt*psiyy + psixx*dpsiyydt - 2*psixy*dpsixydt)
    mspec = ft.invlap*tmpspec
    dzspec = np.zeros(mspec.shape, mspec.dtype)
    dzspec[0,...] = mspec[0,...]/theta1
    dzspec[1,...] = (mspec[1,:]-mspec[0,...])/(theta2-theta1)
    dzspec[0,...] -= dzspec[1,...]
    dzspec = (theta1/grav)*dzspec # convert from exner function to height units (m)
    return ft.spectogrd(dzspec)

def baldiv(ft,vrt,div,dz,dzref=None,f=1.e-4,theta1=300,theta2=330,grav=9.8066,tdrag=5.*86400.,tdiab=20.*86400,nitermax=10,relax=1.0,eps=1.e-9,dzmean=5.e3):
    # computes balanced divergence with iterative algorithm
    # following appendix of https://doi.org/10.1175/1520-0469(1993)050<1519:ACOPAB>2.0.CO;2
    # div on input is initial guess divergence (gridded) - can be set to zero.
    vrtspec = ft.grdtospec(vrt)
    psispec = ft.invlap*vrtspec
    urotspec = -ft.il*psispec; vrotspec = ft.ik*psispec
    urot = ft.spectogrd(urotspec); vrot = ft.spectogrd(vrotspec)
    dzspec = ft.grdtospec(dz)
    dzx = ft.spectogrd(ft.ik*dzspec); dzy = ft.spectogrd(ft.il*dzspec)
    for niter in range(nitermax):
        divspec = ft.grdtospec(div)
        chispec = ft.invlap*divspec
        udivspec = ft.ik*chispec; vdivspec = ft.il*chispec
        udiv = ft.spectogrd(udivspec); vdiv = ft.spectogrd(vdivspec)
        u = urot+udiv; v = vrot+vdiv
        # compute initial guess of vorticity tendency 
        # first, transform fields from spectral space to grid space.
        # diabatic mass flux due to interface relaxation.
        if dzref is not None:
            massflux = (dzref[1] - dz[1])/tdiab
        else:
            massflux = np.zeros((ft.Nt,ft.Nt),ft.dtype)
        # horizontal vorticity flux
        tmp1 = u*(vrt+f); tmp2 = v*(vrt+f)
        # add lower layer drag contribution
        tmp1[0] += v[0]/tdrag
        tmp2[0] += -u[0]/tdrag
        # add diabatic momentum flux contribution
        # (this version averages vertical flux at top
        # and bottom of each layer)
        # same as 'improved' mc2RSW model (DOI: 10.1002/qj.3292)
        tmp1 += 0.5*(v[1]-v[0])*massflux/dz
        tmp2 -= 0.5*(u[1]-u[0])*massflux/dz
        # compute vort flux contributions to vorticity and divergence tend.
        ddivdtspec, dvrtdtspec = ft.getvrtdivspec(tmp1,tmp2)
        dvrtdtspec *= -1
        dvrtdt = ft.spectogrd(dvrtdtspec)
        # infer layer thickness tendency from d/dt of balance eqn.
        ddzdt = nlbalance_tend(ft,vrt,dvrtdt,f=f,theta1=theta1,theta2=theta2,grav=grav)
        # new estimate of divergence from continuity eqn
        divnew = -(1./dz)*(ddzdt + u*dzx + v*dzy)
        divnew = divnew - divnew.mean()
        divdiff = (divnew-div).copy()
        div = div + relax*divdiff
        print(niter,(np.abs(divdiff)).mean(),(np.abs(div)).mean())
        if (np.abs(divdiff)).mean() < eps: break
    return div
        

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

    ntime = -1
    u = nc['u'][ntime]
    v = nc['v'][ntime]
    vrtspec, divspec = ft.getvrtdivspec(u,v)
    vrt = ft.spectogrd(vrtspec); div = ft.spectogrd(divspec)
    dz = nc['dz'][ntime]
    print(dz[0].mean(),dz[1].mean(),nc.zmid,nc.ztop-nc.zmid)

    model = TwoLayer(ft,nc.dt,zmid=nc.zmid,ztop=nc.ztop,tdrag=nc.tdrag,tdiab=nc.tdiab,\
    umax=nc.umax,jetexp=nc.jetexp,theta1=nc.theta1,theta2=nc.theta2,diff_efold=nc.diff_efold)
    nc.close()

    # compute layer thickness tendency
    #tmp1 = u*dz; tmp2 = v*dz
    #tmpspec, ddzdtspec = ft.getvrtdivspec(tmp1,tmp2)
    #ddzdtspec *= -1
    #ddzdt = ft.spectogrd(ddzdtspec)
    ##ddzdt[0] += (model.dzref[1] - dz[1])/model.tdiab
    ##ddzdt[1] -= (model.dzref[1] - dz[1])/model.tdiab
    #dzspec = ft.grdtospec(dz)
    #dzx = ft.spectogrd(ft.ik*dzspec); dzy = ft.spectogrd(ft.il*dzspec)
    ##tmp = ddzdt + u*dzx + v*dzy + div*dz
    ##print(tmp.min(), tmp.max()) # should be zero
    ##raise SystemExit
    #div2 = -(1./dz)*(ddzdt + u*dzx + v*dzy)
    #print(div.min(), div.max())
    #print(div2.min(), div2.max())
    #plt.figure()
    #plt.imshow(div[1],cmap=plt.cm.bwr,vmin=-1.e-5,vmax=1.e-5,interpolation="nearest")
    #plt.colorbar()
    #plt.figure()
    #plt.imshow(div2[1],cmap=plt.cm.bwr,vmin=-1.e-5,vmax=1.e-5,interpolation="nearest")
    #plt.colorbar()
    #plt.show()
    #raise SystemExit
    
    dzbal = nlbalance(ft,vrt,theta1=model.theta1,theta2=model.theta2,dz1mean=dz[0].mean(),dz2mean=dz[1].mean())
    divbal = np.zeros(div.shape, div.dtype) # initialize guess as zero
    divbal = baldiv(ft,vrt,divbal,dzbal,dzref=None,f=model.f,theta1=model.theta1,theta2=model.theta2,\
             grav=model.grav,tdrag=model.tdrag,tdiab=model.tdiab,nitermax=1000,relax=0.02,eps=1.e-8)

    nlevplot = 1
    dzplot = dz[nlevplot]
    dzbalplot = dzbal[nlevplot]
    divplot = div[nlevplot]
    divbalplot = divbal[nlevplot]
    dzmin=0
    dzmax=1.e4
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
        
    dzunbalplot = dzplot-dzbalplot
    print(dzunbalplot.min(), dzunbalplot.max())
    plt.figure()
    plt.imshow(dzplot,cmap=plt.cm.bwr,vmin=dzmin,vmax=dzmax,interpolation="nearest")
    plt.colorbar()
    plt.title('%s layer thickness' % levname)
    plt.figure()
    plt.imshow(dzbalplot,cmap=plt.cm.bwr,vmin=dzmin,vmax=dzmax,interpolation="nearest")
    plt.colorbar()
    plt.title('balanced %s layer thickness' % levname)
    plt.figure()
    plt.imshow(dzunbalplot,cmap=plt.cm.bwr,vmin=-dzunbalmax,vmax=dzunbalmax,interpolation="nearest")
    plt.colorbar()
    plt.title('unbalanced %s layer thickness' % levname)

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
    plt.title('balanced %s layer divergence' % levname)
    plt.figure()
    plt.imshow(divunbalplot,cmap=plt.cm.bwr,vmin=-divunbalmax,vmax=divunbalmax,interpolation="nearest")
    plt.colorbar()
    plt.title('unbalanced %s layer divergence' % levname)

    plt.show()
