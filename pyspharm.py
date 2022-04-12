import numpy as np
# fast spherical harmonic lib from
# https://bitbucket.org/nschaeff/shtns
import shtns

def regrid(spin,spout,datagridin,levs=2):
    # regrid a scalar field
    dataspecin = spin.grdtospec(datagridin)
    if levs == 1:
        dataspecout = np.zeros(spout.nlm, np.complex)
        nmout = 0
        for nm in range(spin.nlm):
            n = spin.degree[nm]
            if n <= spout.ntrunc:
               dataspecout[nmout] = dataspecin[nm]
               nmout += 1
    else:
        dataspecout = np.zeros((levs,spout.nlm), np.complex)
        for lev in range(levs):
            nmout = 0
            for nm in range(spin.nlm):
                n = spin.degree[nm]
                if n <= spout.ntrunc:
                   dataspecout[lev,nmout] = dataspecin[lev,nm]
                   nmout += 1
    datagridout = spout.spectogrd(dataspecout)
    return datagridout

def regriduv(spin,spout,ugridin,vgridin,levs=2):
    # regrid a vector field
    vrtspecin, divspecin = spin.getvrtdivspec(ugridin,vgridin)
    if levs == 1:
        vrtspecout = np.zeros(spout.nlm, np.complex)
        divspecout = np.zeros(spout.nlm, np.complex)
        nmout = 0
        for nm in range(spin.nlm):
            n = spin.degree[nm]
            if n <= spout.ntrunc:
               vrtspecout[nmout] = vrtspecin[nm]
               divspecout[nmout] = divspecin[nm]
               nmout += 1
    else:
        vrtspecout = np.zeros((levs,spout.nlm), np.complex)
        divspecout = np.zeros((levs,spout.nlm), np.complex)
        for lev in range(levs):
            nmout = 0
            for nm in range(spin.nlm):
                n = spin.degree[nm]
                if n <= spout.ntrunc:
                   vrtspecout[lev,nmout] = vrtspecin[lev,nm]
                   divspecout[lev,nmout] = divspecin[lev,nm]
                   nmout += 1
    ugridout,vgridout = spout.getuv(vrtspecout,divspecout)
    return ugridout,vgridout

class Spharmt(object):
    """
    wrapper class for commonly used spectral transform operations in
    atmospheric models

    Jeffrey S. Whitaker <jeffrey.s.whitaker@noaa.gov>
    """
    def __init__(self,nlons,nlats,ntrunc,rsphere,gridtype='gaussian'):
        """initialize
        nlons:  number of longitudes
        nlats:  number of latitudes
        ntrunc: spectral truncation
        rsphere: sphere radius (m)
        gridtype: 'gaussian' (default) or 'regular'"""
        self._shtns = shtns.sht(ntrunc, ntrunc, 1,\
                shtns.sht_fourpi|shtns.SHT_NO_CS_PHASE)
        if gridtype == 'gaussian':
            self._shtns.set_grid(nlats,nlons,shtns.sht_quick_init|shtns.SHT_PHI_CONTIGUOUS,1.e-8)
        elif gridtype == 'regular':
            self._shtns.set_grid(nlats,nlons,shtns.sht_reg_dct|shtns.SHT_PHI_CONTIGUOUS,1.e-8)
        #self._shtns.print_info()
        self.lats = np.arcsin(self._shtns.cos_theta)
        self.lons = (2.*np.pi/nlons)*np.arange(nlons)
        self.nlons = nlons
        self.nlats = nlats
        self.ntrunc = ntrunc
        self.nlm = self._shtns.nlm
        self.degree = self._shtns.l
        self.order = self._shtns.m
        if gridtype == 'gaussian':
            self.gauwts =\
            np.concatenate((self._shtns.gauss_wts(),self._shtns.gauss_wts()[::-1]))
        else:
            self.gauwts = None
        self.gridtype = gridtype
        self.lap = -self.degree*(self.degree+1.0).astype(np.complex)
        self.invlap = np.zeros(self.lap.shape, self.lap.dtype)
        self.invlap[1:] = 1./self.lap[1:]
        self.rsphere = rsphere
        self.lap = self.lap/rsphere**2
        self.invlap = self.invlap*rsphere**2
    def smooth(self,data,smoothfact):
        """smooth with gaussian spectral smoother"""
        dataspec = self.grdtospec(data)
        smoothspec = np.exp(self.lap/(smoothfact*(smoothfact+1.)))
        return self.spectogrd(smoothspec*dataspec)
    def grdtospec(self,data):
        """compute spectral coefficients from gridded data"""
        data = np.ascontiguousarray(data, dtype=np.float)
        if data.ndim == 2:
            dataspec = np.empty(self.nlm, dtype=np.complex)
            self._shtns.spat_to_SH(data, dataspec)
        elif data.ndim == 3:
            dataspec = np.empty((data.shape[0],self.nlm), dtype=np.complex)
            for k,d in enumerate(data):
                self._shtns.spat_to_SH(d, dataspec[k])
        else:
            raise IndexError('data must be 2d or 3d')
        return dataspec
    def spectogrd(self,dataspec):
        """compute gridded data from spectral coefficients"""
        dataspec = np.ascontiguousarray(dataspec, dtype=np.complex)
        if dataspec.ndim == 1:
            data = np.empty((self.nlats,self.nlons), dtype=np.float)
            self._shtns.SH_to_spat(dataspec, data)
        elif dataspec.ndim == 2:
            data = np.empty((dataspec.shape[0],self.nlats,self.nlons), dtype=np.float)
            for k,d in enumerate(dataspec):
                self._shtns.SH_to_spat(d, data[k])
        else:
            raise IndexError('dataspec must be 1d or 2d')
        return data
    def getuv(self,vrtspec,divspec):
        """compute wind vector from spectral coeffs of vorticity and divergence"""
        vrtspec = np.ascontiguousarray(vrtspec, dtype=np.complex)
        divspec = np.ascontiguousarray(divspec, dtype=np.complex)
        if vrtspec.ndim == 1:
            u = np.empty((self.nlats,self.nlons), dtype=np.float)
            v = np.empty((self.nlats,self.nlons), dtype=np.float)
            self._shtns.SHsphtor_to_spat((self.invlap/self.rsphere)*vrtspec,\
               (self.invlap/self.rsphere)*divspec, u, v)
        elif vrtspec.ndim == 2:
            u = np.empty((vrtspec.shape[0],self.nlats,self.nlons), dtype=np.float)
            v = np.empty((vrtspec.shape[0],self.nlats,self.nlons), dtype=np.float)
            for k,vrt in enumerate(vrtspec):
                div = divspec[k]
                self._shtns.SHsphtor_to_spat((self.invlap/self.rsphere)*vrt,\
                   (self.invlap/self.rsphere)*div, u[k], v[k])
        else:
            raise IndexError('vrtspec,divspec must be 1d or 2d')
        return u,v
    def getvrtdivspec(self,u,v):
        """compute spectral coeffs of vorticity and divergence from wind vector"""
        u = np.ascontiguousarray(u, dtype=np.float)
        v = np.ascontiguousarray(v, dtype=np.float)
        if u.ndim == 2:
            vrtspec = np.empty(self.nlm, dtype=np.complex)
            divspec = np.empty(self.nlm, dtype=np.complex)
            self._shtns.spat_to_SHsphtor(u, v, vrtspec, divspec)
        elif u.ndim == 3:
            vrtspec = np.empty((u.shape[0],self.nlm), dtype=np.complex)
            divspec = np.empty((u.shape[0],self.nlm), dtype=np.complex)
            for k,uu in enumerate(u):
                vv = v[k]
                self._shtns.spat_to_SHsphtor(uu, vv, vrtspec[k], divspec[k])
        else:
            raise IndexError('u,v must be 2d or 3d')
        return self.lap*self.rsphere*vrtspec, self.lap*self.rsphere*divspec
    def getgrad(self,dataspec):
        """compute gradient vector from spectral coeffs"""
        #divspec = np.ascontiguousarray(dataspec, dtype=np.complex)
        #vrtspec = np.zeros(divspec.shape, dtype=np.complex)
        #if dataspec.ndim == 1:
        #    gradx = np.empty((self.nlats,self.nlons), dtype=np.float)
        #    grady = np.empty((self.nlats,self.nlons), dtype=np.float)
        #    self._shtns.SHsphtor_to_spat(vrtspec,divspec,gradx,grady)
        #elif dataspec.ndim == 2:
        #    gradx = np.empty((dataspec.shape[0],self.nlats,self.nlons), dtype=np.float)
        #    grady = np.empty((dataspec.shape[0],self.nlats,self.nlons), dtype=np.float)
        #    for k,div in enumerate(divspec):
        #        vrt = vrtspec[k] 
        #        self._shtns.SHsphtor_to_spat(vrt,div,gradx[k],grady[k])
        #else:
        #    raise IndexError('dataspec must be 1d or 2d')
        dataspec = np.ascontiguousarray(dataspec, dtype=np.complex)
        if dataspec.ndim == 1:
            grady,gradx = self._shtns.synth_grad(dataspec)
        elif dataspec.ndim == 2:
            gradx = np.empty((dataspec.shape[0],self.nlats,self.nlons), dtype=np.float)
            grady = np.empty((dataspec.shape[0],self.nlats,self.nlons), dtype=np.float)
            for k,spec in enumerate(dataspec):
                grady[k],gradx[k] = self._shtns.synth_grad(spec)
        else:
            raise IndexError('dataspec must be 1d or 2d')
        return gradx/self.rsphere, -grady/self.rsphere
