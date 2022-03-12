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
dzbal, vrtbal, divbal = model.pvinvert(pv,dzin=dz,\
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
dzbal,divbal = model.nlbalance(vrtspec,div=True,dz1mean=dz[0].mean(),dz2mean=dz[1].mean(),\
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
