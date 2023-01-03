import numpy as np
from twolayer import TwoLayer as TwoLayer_base

class TwoLayer(TwoLayer_base):

    def gettend(self,vrtspec,divspec,dzspec,masstend_diag=False):
        ub = self.ub; vb = self.vb
        vrtb = self.vrtb; dzb = self.dzb
        u,v = self.ft.getuv(vrtspec,divspec)
        dz = self.ft.spectogrd(dzspec)
        vrt = self.ft.spectogrd(vrtspec)
        # horizontal vorticity flux
        tmp1 = u*(vrtb+self.f) + ub*vrt
        tmp2 = v*(vrtb+self.f) + vb*vrt
        # add linear layer drag contribution
        tmp1 += v/self.tdrag[:,np.newaxis,np.newaxis]
        tmp2 += -u/self.tdrag[:,np.newaxis,np.newaxis]
        # compute vort flux contributions to vorticity and divergence tend.
        ddivdtspec, dvrtdtspec = self.ft.getvrtdivspec(tmp1,tmp2)
        ddivdt = self.ft.spectogrd(ddivdtspec)
        #dvrtdtspec *= -1
        # hyperdiffusion of vorticity and divergence
        #dvrtdtspec += self.hyperdiff*vrtspec
        dvrtdtspec = np.zeros_like(ddivdtspec) # hold vorticity constant
        # diabatic mass flux due to interface relaxation.
        massflux = -dz[1]/self.tdiab
        # horizontal mass flux contribution to continuity
        tmpg1 = ub*dz + u*dzb; tmpg2 = vb*dz + v*dzb
        tmpspec, ddzdtspec = self.ft.getvrtdivspec(tmpg1,tmpg2)
        if masstend_diag:
            # mean abs total mass tend (meters/hour)
            totmassdiv = (self.ft.spectogrd(ddzdtspec)).sum(axis=0)
            #print(self.t,totmassdiv.min(), totmassdiv.max())
            self.masstendvar = 3600.*np.abs(totmassdiv).mean()
            print(self.t,self.masstendvar)
        ddzdtspec *= -1
        # diabatic mass flux contribution to continuity
        tmpspec = self.ft.grdtospec(massflux)
        ddzdtspec[0] -= tmpspec; ddzdtspec[1] += tmpspec
        # pressure gradient force contribution to divergence tend
        mstrm = np.empty(dz.shape, dtype=self.dtype) # montgomery streamfunction
        mstrm[0] = self.grav*(dz[0] + dz[1])
        mstrm[1] = mstrm[0] + (self.grav*self.delth/self.theta1)*dz[1]
        ddivdtspec += -self.ft.lap*self.ft.grdtospec(mstrm+(ub*u + vb*v))
        # hyperdiffusion of divergence
        # extra laplacian diffusion of divergence to suppress gravity waves
        ddivdtspec += self.hyperdiff*divspec + self.divlapdiff*divspec
        return dvrtdtspec,ddivdtspec,ddzdtspec
