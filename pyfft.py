from __future__ import print_function
import numpy as np

try:  # pyfftw is *much* faster
    from pyfftw.interfaces import numpy_fft
    rfft2 = numpy_fft.rfft2
    irfft2 = numpy_fft.irfft2
except ImportError:  # fall back on numpy fft.
    print("# WARNING: using numpy fft (install mkl_fft or pyfftw for better performance)...")

    def rfft2(*args, **kwargs):
        kwargs.pop("threads", None)
        return np.fft.rfft2(*args, **kwargs)

    def irfft2(*args, **kwargs):
        kwargs.pop("threads", None)
        return np.fft.irfft2(*args, **kwargs)

class Fouriert(object):
    """
    wrapper class for commonly used spectral transform operations in
    atmospheric models

    Jeffrey S. Whitaker <jeffrey.s.whitaker@noaa.gov>
    """
    def __init__(self,N,L,threads=1,dealias=True):
        """initialize
        N: number of grid points (spectral truncation x 2)
        L: domain size"""
        self.N = N
        if dealias:
            self.Nt = 3*N//2
        else:
            self.Nt = N
        self.dealias = dealias
        self.L = L
        self.threads = threads
        # spectral stuff
        k = (N * np.fft.fftfreq(N))[0 : (N // 2) + 1]
        l = N * np.fft.fftfreq(N)
        k, l = np.meshgrid(k, l)
        k = k.astype(np.float32)
        l = l.astype(np.float32)
        # dimensionalize wavenumbers.
        pi = np.array(np.pi, np.float32)
        k = 2.0 * pi * k / self.L
        l = 2.0 * pi * l / self.L
        ksqlsq = k ** 2 + l ** 2
        self.k = k
        self.l = l
        self.ksqlsq = ksqlsq
        self.ik = (1.0j * k).astype(np.complex64)
        self.il = (1.0j * l).astype(np.complex64)
        self.lap = -ksqlsq.astype(np.complex64)
        self.invlap = np.where(ksqlsq > 0, 1./self.lap, 0.)
    def grdtospec(self,data):
        """compute spectral coefficients from gridded data"""
        # if dealias==True, spectral data is truncated to 2/3 size
        # size 2, self.N, self.N // 2 + 1
        dataspec = rfft2(data, threads=self.threads)
        if self.dealias: 
            return self.spectrunc(dataspec)
        else:
            return dataspec
    def spectogrd(self,dataspec):
        """compute gridded data from spectral coefficients"""
        # if dealias==True, data returned on 3/2 grid
        # dataspec padded with zeros to 2, 3 * self.N // 2, 3 * self.N // 4 + 1
        # data returned on 2, 3*self.N//2, 3*self.N//2
        if self.dealias:
            dataspec_tmp = self.specpad(dataspec)
        else:
            dataspec_tmp = dataspec
        return irfft2(dataspec_tmp, threads=self.threads)
    def getuv(self,vrtspec,divspec):
        """compute wind vector from spectral coeffs of vorticity and divergence"""
        psispec = self.invlap*vrtspec
        chispec = self.invlap*divspec
        psix, psiy = self.getgrad(psispec)
        chix, chiy = self.getgrad(chispec)
        u = -psiy + chix
        v = psix + chiy
        return u,v
    def getvrtdivspec(self,u,v):
        """compute spectral coeffs of vorticity and divergence from wind vector"""
        uspec = self.grdtospec(u); vspec = self.grdtospec(v)
        vrtspec = self.ik*vspec  - self.il*uspec
        divspec = self.ik*uspec  + self.il*vspec
        return vrtspec, divspec
    def getgrad(self, dataspec):
        return self.spectogrd(self.ik*dataspec), self.spectogrd(self.il*dataspec)
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
