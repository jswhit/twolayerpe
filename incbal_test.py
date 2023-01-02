import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
import numpy as np
from pyfft import Fouriert
from twolayer_bal import TwoLayer

from netCDF4 import Dataset
filename = sys.argv[1]
nc = Dataset(filename)

ft = Fouriert(nc.N,nc.L,precision=str(nc['dz_a'].dtype))
model=TwoLayer(ft,nc.dt,theta1=nc.theta1,theta2=nc.theta2,f=nc.f,\
zmid=nc.zmid,ztop=nc.ztop,diff_efold=nc.diff_efold,diff_order=nc.diff_order,tdrag1=nc.tdrag[0],tdrag2=nc.tdrag[1],tdiab=nc.tdiab,umax=nc.umax)

ntime = int(sys.argv[2])

u = nc['u_a'][ntime]
v = nc['v_a'][ntime]
dz = nc['dz_a'][ntime]
uensmean = u.mean(axis=0)
vensmean = v.mean(axis=0)
dzensmean = dz.mean(axis=0)
print('area mean layer thickness = ',dzensmean[0].mean(), dzensmean[1].mean())
upert = u-uensmean; vpert = v-vensmean; dzpert = dz-dzensmean
vrtspec_ensmean, divspec_ensmean = ft.getvrtdivspec(uensmean,vensmean)

nmem = 1
vrtspec,divspec = ft.getvrtdivspec(upert[nmem],vpert[nmem])
divpert = ft.spectogrd(divspec)
dzpert_bal,divpert_bal = model.nlbalinc(vrtspec_ensmean,divspec_ensmean,dzensmean,vrtspec,linbal=False,baldiv=True)

print(dzpert[nmem].min(), dzpert[nmem].max())
print(dzpert_bal.min(), dzpert_bal.max())
print(divpert.min(), divpert.max())
print(divpert_bal.min(), divpert_bal.max())

nlevplot = 1
levname = 'upper'

dzplot = dzpert[nmem,nlevplot,...]
dzbalplot = dzpert_bal[nlevplot]
divplot = divpert[nlevplot]
divbalplot = divpert_bal[nlevplot]
dzunbalplot = dzplot-dzbalplot
divunbalplot = divplot-divbalplot
print(dzunbalplot.min(), dzunbalplot.max())
print(divunbalplot.min(), divunbalplot.max())

dzmin = -100; dzmax = 100
plt.figure(figsize=(6,15))
plt.subplot(3,1,1)
plt.imshow(dzplot,cmap=plt.cm.bwr,vmin=dzmin,vmax=dzmax,interpolation="nearest")
plt.colorbar()
plt.title('%s layer thickness pert' % levname)
plt.subplot(3,1,2)
plt.imshow(dzbalplot,cmap=plt.cm.bwr,vmin=dzmin,vmax=dzmax,interpolation="nearest")
plt.colorbar()
plt.title('balanced %s layer thickness pert' % levname)
plt.subplot(3,1,3)
dzmin = -10; dzmax = 10
plt.imshow(dzunbalplot,cmap=plt.cm.bwr,vmin=dzmin,vmax=dzmax,interpolation="nearest")
plt.colorbar()
plt.title('unbalanced %s layer thickness pert' % levname)
plt.savefig('inbcal_test_dz.png')

divmin = -5.e-7; divmax = 5.e-7
plt.figure(figsize=(6,15))
plt.subplot(3,1,1)
plt.imshow(divplot,cmap=plt.cm.bwr,vmin=divmin,vmax=divmax,interpolation="nearest")
plt.colorbar()
plt.title('%s layer thickness pert' % levname)
plt.subplot(3,1,2)
plt.imshow(divbalplot,cmap=plt.cm.bwr,vmin=divmin,vmax=divmax,interpolation="nearest")
plt.colorbar()
plt.title('balanced %s layer thickness pert' % levname)
plt.subplot(3,1,3)
divmin = -5.e-7; divmax = 5.e-7
plt.imshow(divunbalplot,cmap=plt.cm.bwr,vmin=divmin,vmax=divmax,interpolation="nearest")
plt.colorbar()
plt.title('unbalanced %s layer thickness pert' % levname)
plt.savefig('inbcal_test_div.png')
