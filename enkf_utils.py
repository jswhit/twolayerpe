import numpy as np
from scipy.linalg import eigh, cho_solve, cho_factor

# function definitions.


def cartdist(x1, y1, x2, y2, xmax, ymax):
    """cartesian distance on doubly periodic plane"""
    dx = np.abs(x1 - x2)
    dy = np.abs(y1 - y2)
    dx = np.where(dx > 0.5 * xmax, xmax - dx, dx)
    dy = np.where(dy > 0.5 * ymax, ymax - dy, dy)
    return np.sqrt(dx ** 2 + dy ** 2)


def gaspcohn(r):
    """
    Gaspari-Cohn taper function.
    very close to exp(-(r/c)**2), where c = sqrt(0.15)
    r should be >0 and normalized so taper = 0 at r = 1
    """
    rr = 2.0 * r
    rr += 1.0e-13  # avoid divide by zero warnings from numpy
    taper = np.where(
        r <= 0.5,
        (((-0.25 * rr + 0.5) * rr + 0.625) * rr - 5.0 / 3.0) * rr ** 2 + 1.0,
        np.zeros(r.shape, r.dtype),
    )
    taper = np.where(
        np.logical_and(r > 0.5, r < 1.0),
        ((((rr / 12.0 - 0.5) * rr + 0.625) * rr + 5.0 / 3.0) * rr - 5.0) * rr
        + 4.0
        - 2.0 / (3.0 * rr),
        taper,
    )
    return taper

def letkf_kernel(xens,hxens,obs,oberrs,covlocal):
    nanals, nlevs = xens.shape
    nobs = len(obs)
    xmean = xens.mean(axis=0)
    xprime = xens - xmean
    hxmean = hxens.mean(axis=0)
    hxprime = hxens - hxmean
    def calcwts(hx, Rinv, ominusf):
        YbRinv = np.dot(hx, Rinv)
        pa = (nanals - 1) * np.eye(nanals) + np.dot(YbRinv, hx.T)
        evals, eigs = np.linalg.eigh(pa)
        painv = np.dot(np.dot(eigs, np.diag(np.sqrt(1.0 / evals))), eigs.T)
        tmp = np.dot(np.dot(np.dot(painv, painv.T), YbRinv), ominusf)
        return np.sqrt(nanals - 1) * painv + tmp[:, np.newaxis]
    for k in range(nlevs):
        mask = covlocal > 1.e-10
        Rinv = np.diag(covlocal[mask] / oberrs[mask])
        ominusf = (obs-hxmean)[mask]
        wts = calcwts(hxprime[:, mask], Rinv, ominusf)
        xens[:, k] = xmean[k] + np.dot(wts.T, xprime[:, k])
    return xens

def letkf_update(xens,hxens,obs,oberrs,covlocal):
    """letkf method"""
    ndim = xens.shape[-1]
    for n in range(ndim): # horizontal grid (TODO: parallelize this embarassingly parallel loop)
        xens[:,:,n] = letkf_kernel(xens[:,:,n],hxens,obs,oberrs,covlocal[:,n])
    return xens

def serial_update(xens, hxens, obs, oberrs, covlocal, obcovlocal):
    """serial potter method"""
    nanals, nlevs, ndim = xens.shape
    nobs = len(obs)
    xmean = xens.mean(axis=0)
    xprime = xens - xmean
    hxmean = hxens.mean(axis=0)
    hxprime = hxens - hxmean
    for nob, ob, oberr in zip(np.arange(nobs), obs, oberrs):
        ominusf = ob - hxmean[nob].copy()
        hxens = hxprime[:, nob].copy().reshape((nanals, 1))
        hpbht = (hxens ** 2).sum() / (nanals - 1)
        gainfact = (
            (hpbht + oberr)
            / hpbht
            * (1.0 - np.sqrt(oberr / (hpbht + oberr)))
        )
        # state space update
        # only update points closer than localization radius to ob
        mask = covlocal[nob, :] > 1.e-10
        for k in range(nlevs):
            pbht = (xprime[:, k, mask].T * hxens[:, 0]).sum(axis=1) / float(
                nanals - 1
            )
            kfgain = covlocal[nob, mask] * pbht / (hpbht + oberr)
            xmean[k, mask] = xmean[k, mask] + kfgain * ominusf
            xprime[:, k, mask] = xprime[:, k, mask] - gainfact * kfgain * hxens
        # observation space update
        # only update obs within localization radius
        mask = obcovlocal[nob, :] > 1.e-10
        pbht = (hxprime[:, mask].T * hxens[:, 0]).sum(axis=1) / float(
            nanals - 1
        )
        kfgain = obcovlocal[nob, mask] * pbht / (hpbht + oberr)
        hxmean[mask] = hxmean[mask] + kfgain * ominusf
        hxprime[:, mask] = (
            hxprime[:, mask] - gainfact * kfgain * hxens
        )
    return xmean + xprime
