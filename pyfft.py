import numpy as np
import pyfftw

class Fouriert(object):
    """
    wrapper class for commonly used spectral transform operations in
    atmospheric models

    Jeffrey S. Whitaker <jeffrey.s.whitaker@noaa.gov>
    """
    def __init__(self,N,L,threads=1,precision='single'):
        """initialize
        N: number of grid points (spectral truncation x 2)
        L: domain size"""
        self.L = L
        self.threads = threads
        self.N = N
        self.Nt = 3*N//2
        if precision == 'single':
            dtype = 'float32'
            dtypec = 'complex64'
        else:
            dtype = 'float64'
            dtypec = 'complex128'
        self.precision = precision
        # set up pyfftw objects for transforms
        self.rfft2=pyfftw.builders.rfft2(pyfftw.empty_aligned((2,self.Nt,self.Nt), dtype=dtype),\
                                          axes=(-2, -1), threads=threads, planner_effort='FFTW_ESTIMATE')
        self.irfft2=pyfftw.builders.irfft2(pyfftw.empty_aligned((2,self.Nt,self.Nt//2+1), dtype=dtypec),\
                                          axes=(-2, -1), threads=threads, planner_effort='FFTW_ESTIMATE')
        self.rfft2_2d=pyfftw.builders.rfft2(pyfftw.empty_aligned((self.Nt,self.Nt), dtype=dtype),\
                                          axes=(-2, -1), threads=threads, planner_effort='FFTW_ESTIMATE')
        self.irfft2_2d=pyfftw.builders.irfft2(pyfftw.empty_aligned((self.Nt,self.Nt//2+1), dtype=dtypec),\
                                          axes=(-2, -1), threads=threads, planner_effort='FFTW_ESTIMATE')
        # spectral stuff
        dk = 2.*np.pi/self.L
        #k =  dk*np.arange(0.,self.N//2+1)
        #l =  dk*np.append( np.arange(0.,self.N//2),np.arange(-self.N//2,0.) )
        k = dk*(N * np.fft.fftfreq(N))[0 : (N // 2) + 1] # last freq is negative, unlike above
        l = dk*N * np.fft.fftfreq(N)
        k, l = np.meshgrid(k, l)
        self.k = k.astype(dtype)
        self.l = l.astype(dtype)
        ksqlsq = self.k ** 2 + self.l ** 2
        self.ksqlsq = ksqlsq
        self.ik = (1.0j * k).astype(dtypec)
        self.il = (1.0j * l).astype(dtypec)
        self.lap = -ksqlsq.astype(dtypec)
        lapnonzero = self.lap != 0.
        self.invlap = np.zeros_like(self.lap)
        self.invlap[lapnonzero] = 1./self.lap[lapnonzero]
    def grdtospec(self,data):
        """compute spectral coefficients from gridded data"""
        if data.ndim==2:
            dataspec = self.rfft2_2d(data)
        else:
            dataspec = self.rfft2(data)
        return self.spectrunc(dataspec)
    def spectogrd(self,dataspec):
        """compute gridded data from spectral coefficients"""
        dataspec_tmp = self.specpad(dataspec)
        if dataspec_tmp.ndim==2:
            data =  self.irfft2_2d(dataspec_tmp)
        else:
            data =  self.irfft2(dataspec_tmp)
        return np.array(data,copy=True)
    def getuv(self,vrtspec,divspec):
        """compute wind vector from spectral coeffs of vorticity and divergence"""
        psispec = self.invlap*vrtspec
        chispec = self.invlap*divspec
        uspec = -self.il*psispec + self.ik*chispec
        vspec = self.ik*psispec + self.il*chispec
        return self.spectogrd(uspec), self.spectogrd(vspec)
    def getvrtdivspec(self,u,v):
        """compute spectral coeffs of vorticity and divergence from wind vector"""
        uspec = self.grdtospec(u); vspec = self.grdtospec(v)
        vrtspec = self.ik*vspec  - self.il*uspec
        divspec = self.ik*uspec  + self.il*vspec
        return vrtspec, divspec
    def specpad(self, specarr):
        # pad spectral arrays with zeros to get
        # interpolation to 3/2 larger grid using inverse fft.
        # take care of normalization factor for inverse transform.
        if specarr.ndim == 3:
            specarr_pad = np.zeros((2, self.Nt, self.Nt// 2 + 1), specarr.dtype)
            specarr_pad[:, 0 : self.N // 2, 0 : self.N // 2] = (
                specarr[:, 0 : self.N // 2, 0 : self.N // 2]
            )
            specarr_pad[:, -self.N // 2 :, 0 : self.N // 2] = (
                specarr[:, -self.N // 2 :, 0 : self.N // 2]
            )
            # include negative Nyquist frequency.
            specarr_pad[:, 0 : self.N // 2, self.N // 2] = np.conjugate(
                specarr[:, 0 : self.N // 2, -1]
            )
            specarr_pad[:, -self.N // 2 :, self.N // 2] = np.conjugate(
                specarr[:, -self.N // 2 :, -1]
            )
        elif specarr.ndim==2:
            specarr_pad = np.zeros((self.Nt, self.Nt// 2 + 1), specarr.dtype)
            specarr_pad[0 : self.N // 2, 0 : self.N // 2] = (
                specarr[0 : self.N // 2, 0 : self.N // 2]
            )
            specarr_pad[-self.N // 2 :, 0 : self.N // 2] = (
                specarr[-self.N // 2 :, 0 : self.N // 2]
            )
            # include negative Nyquist frequency.
            specarr_pad[0 : self.N // 2, self.N // 2] = np.conjugate(
                specarr[0 : self.N // 2, -1]
            )
            specarr_pad[-self.N // 2 :, self.N // 2] = np.conjugate(
                specarr[-self.N // 2 :, -1]
            )
        else:
            raise IndexError('specarr must be 2d or 3d')
        return 2.25*specarr_pad
    def spectrunc(self, specarr):
        # truncate spectral array using 2/3 rule.
        if specarr.ndim == 3:
            specarr_trunc = np.zeros((2, self.N, self.N // 2 + 1), specarr.dtype)
            specarr_trunc[:, 0 : self.N // 2, 0 : self.N // 2] = specarr[
                :, 0 : self.N // 2, 0 : self.N // 2
            ]
            specarr_trunc[:, -self.N // 2 :, 0 : self.N // 2] = specarr[
                :, -self.N // 2 :, 0 : self.N // 2
            ]
        elif specarr.ndim == 2:
            specarr_trunc = np.zeros((self.N, self.N // 2 + 1), specarr.dtype)
            specarr_trunc[0 : self.N // 2, 0 : self.N // 2] = specarr[
                0 : self.N // 2, 0 : self.N // 2
            ]
            specarr_trunc[-self.N // 2 :, 0 : self.N // 2] = specarr[
                -self.N // 2 :, 0 : self.N // 2
            ]
        else:
            raise IndexError('specarr must be 2d or 3d')
        return specarr_trunc/2.25
