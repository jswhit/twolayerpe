def nlbalance(ft,u,vg,orog=None,f=1.e-4,theta1=300,theta2=330,grav=9.8066,dz1mean=5.e3,dz2mean=5.e3):
    # ft: Fourier transform object
    # f: coriolis param
    # grav: gravity
    # theta1,theta2: pot temp in each layer
    # dz1mean, dz2mean: area mean layer thicknesses (state of rest)
    # u,vg: winds in each layer
    # returns dz, layer thickness of each layer
    # solve nonlinear balance eqn to get layer thickness given winds
    dzspec = np.zeros((2,ft.N,ft.N//2+1), ft.dtypec)
    vrtspec, divspec = ft.getvrtdivspec(u,vg)
    vrt = ft.spectogrd(vrtspec)
    tmp1 = u*(vrt+f); tmp2 = vg*(vrt+f)
    tmpspec1, tmpspec2 = ft.getvrtdivspec(tmp1,tmp2)
    tmpspec2 = ft.grdtospec(0.5*(u**2+vg**2))
    mspec = ft.invlap*tmpspec1 - tmpspec2
    mstream = ft.spectogrd(mspec)
    dzspec[0,...] = mspec[0,...]/theta1
    dzspec[1,...] = (mspec[1,:]-mspec[0,...])/(theta2-theta1)
    dzspec[0,...] = dzspec[0,...] - dzspec[1,...]
    dzspec = (theta1/grav)*dzspec # convert from exner function to height units (m)
    # set area mean in grid space to state of rest value
    dz = ft.spectogrd(dzspec)
    dz[0,...] = dz[0,...] - dz[0,...].mean() + dz1mean
    dz[1,...] = dz[1,...] - dz[1,...].mean() + dz2mean
    return dz

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('qt5agg')
    import matplotlib.pyplot as plt
    import sys
    import numpy as np
    from pyfft import Fouriert

    from netCDF4 import Dataset
    filename = sys.argv[1]
    nc = Dataset(filename)

    x = nc.variables['x'][:]
    y = nc.variables['y'][:]
    x, y = np.meshgrid(x, y)
    theta1 = nc.theta1
    theta2 = nc.theta2
    f = nc.f
    grav = nc.grav
    zmid = nc.zmid
    ztop = nc.ztop
    N = nc.N
    Nt = nc.Nt
    L = nc.L
    ntime = -1
    u = nc['u'][ntime]
    v = nc['v'][ntime]
    dz = nc['dz'][ntime]
    print(dz[0].mean(),dz[1].mean(),zmid,ztop-zmid)
    
    ft = Fouriert(N,L)
    dzbal = nlbalance(ft,u,v,theta2=theta2)

    nlevplot = -1
    dzplot = dz[nlevplot]
    dzbalplot = dzbal[nlevplot]
    dzmin=0
    dzmax=1.e4
    dzunbalmax=40
    if nlevplot == 1:
        levname='upper'
    elif nlevplot == 0:
        levname='lower'
    elif nlevplot < 0:
        levname='total'
        dzplot = ztop - dz.sum(axis=0)
        dzbalplot = ztop - dzbal.sum(axis=0)
        dzmin = -200
        dzmax = 200.
        dzunbalmax = 2
        
    dzunbalplot = dzplot-dzbalplot
    print(dzunbalplot.min(), dzunbalplot.max())
    plt.figure()
    plt.imshow(dzplot,cmap=plt.cm.bwr,vmin=dzmin,vmax=dzmax,interpolation="nearest")
    plt.title('%s layer thickness' % levname)
    plt.figure()
    plt.imshow(dzbalplot,cmap=plt.cm.bwr,vmin=dzmin,vmax=dzmax,interpolation="nearest")
    plt.title('balanced %s layer thickness' % levname)
    plt.figure()
    plt.imshow(dzunbalplot,cmap=plt.cm.bwr,vmin=-dzunbalmax,vmax=dzunbalmax,interpolation="nearest")
    plt.title('unbalanced %s layer thickness' % levname)
    plt.show()
