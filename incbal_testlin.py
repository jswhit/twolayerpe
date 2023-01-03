import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
import numpy as np
from pyfft import Fouriert
from twolayer_lin import TwoLayer
from netCDF4 import Dataset

filename = sys.argv[1]
nc = Dataset(filename)

div2_diff_efold=nc.dt

ft = Fouriert(nc.N,nc.L,precision=str(nc['dz_a'].dtype))
model=TwoLayer(ft,nc.dt,theta1=nc.theta1,theta2=nc.theta2,f=nc.f,\
zmid=nc.zmid,ztop=nc.ztop,diff_efold=nc.diff_efold,diff_order=nc.diff_order,tdrag1=nc.tdrag[0],tdrag2=nc.tdrag[1],tdiab=nc.tdiab,umax=nc.umax,\
div2_diff_efold=div2_diff_efold)

ntime = 1
Nt = model.ft.Nt
u = nc['u_b'][ntime]
v = nc['v_b'][ntime]
dz = nc['dz_b'][ntime]
ub = u.mean(axis=0)
vb = v.mean(axis=0)
dzb = dz.mean(axis=0)
vrtspec_ensmean, divspec_ensmean = ft.getvrtdivspec(ub,vb)
vrtb = ft.spectogrd(vrtspec_ensmean)
uinc = (nc['u_a'][ntime]).mean(axis=0)-ub
vinc = (nc['v_a'][ntime]).mean(axis=0)-vb
dzinc = (nc['dz_a'][ntime]).mean(axis=0)-dzb
vrtspec, divspec = ft.getvrtdivspec(uinc,vinc)
divinc = ft.spectogrd(divspec)
vrtinc = ft.spectogrd(vrtspec)
dzspec = ft.grdtospec(dzinc)
model.ub = ub; model.vb = vb
model.vrtb = vrtb; model.dzb = dzb

tmax = 10.*86400.
nsteps = int(tmax/model.dt) # number of time steps to run
for n in range(nsteps):
    vrtspec, divspec, dzspec = model.rk4step(vrtspec, divspec, dzspec)
dzpert_bal = model.ft.spectogrd(dzspec)
divpert_bal = model.ft.spectogrd(divspec)

nlevplot = 1
levname = 'upper'

dzplot = dzinc[nlevplot,...]
dzbalplot = dzpert_bal[nlevplot,...]
divplot = divinc[nlevplot,...]
divbalplot = divpert_bal[nlevplot,...]
dzunbalplot = dzplot-dzbalplot
divunbalplot = divplot-divbalplot
print(dzbalplot.min(), dzbalplot.max())
print(divbalplot.min(), divbalplot.max())
print(dzunbalplot.min(), dzunbalplot.max())
print(divunbalplot.min(), divunbalplot.max())

dzmin = -200; dzmax = 200
plt.figure(figsize=(6,15))
plt.subplot(3,1,1)
plt.imshow(dzplot,cmap=plt.cm.bwr,vmin=dzmin,vmax=dzmax,interpolation="nearest")
plt.colorbar()
plt.title('%s layer thickness inc' % levname)
plt.subplot(3,1,2)
plt.imshow(dzbalplot,cmap=plt.cm.bwr,vmin=dzmin,vmax=dzmax,interpolation="nearest")
plt.colorbar()
plt.title('balanced %s layer thickness inc' % levname)
plt.subplot(3,1,3)
dzmin = -200; dzmax = 200
plt.imshow(dzunbalplot,cmap=plt.cm.bwr,vmin=dzmin,vmax=dzmax,interpolation="nearest")
plt.colorbar()
plt.title('unbalanced %s layer thickness inc' % levname)
plt.savefig('inbcal_test2_dz.png')

divmin = -1.e-6; divmax = 1.e-6
plt.figure(figsize=(6,15))
plt.subplot(3,1,1)
plt.imshow(divplot,cmap=plt.cm.bwr,vmin=divmin,vmax=divmax,interpolation="nearest")
plt.colorbar()
plt.title('%s layer divergence inc' % levname)
plt.subplot(3,1,2)
plt.imshow(divbalplot,cmap=plt.cm.bwr,vmin=divmin,vmax=divmax,interpolation="nearest")
plt.colorbar()
plt.title('balanced %s layer divergence inc' % levname)
plt.subplot(3,1,3)
divmin = -1.e-6; divmax = 1.e-6
plt.imshow(divunbalplot,cmap=plt.cm.bwr,vmin=divmin,vmax=divmax,interpolation="nearest")
plt.colorbar()
plt.title('unbalanced %s layer divergence inc' % levname)
plt.savefig('inbcal_test2_div.png')
