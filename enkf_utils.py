import numpy as np
from scipy.linalg import eigh
from joblib import Parallel, delayed

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

def letkf_kernel(xens,hxens,obs,oberrs,covlocal,nlevs_update=None):
    nanals, nlevs = xens.shape
    if nlevs_update is not None:
        # only update 1st nlevs_update 2d fields.
        nlevs = nlevs_update
    nobs = len(obs)
    xmean = xens.mean(axis=0)
    xprime = xens - xmean
    hxmean = hxens.mean(axis=0)
    hxprime = hxens - hxmean
    def calcwts(hx, Rinv, ominusf):
        YbRinv = np.dot(hx, Rinv)
        pa = (nanals - 1) * np.eye(nanals) + np.dot(YbRinv, hx.T)
        evals, eigs = eigh(pa)
        painv = np.dot(np.dot(eigs, np.diag(np.sqrt(1.0 / evals))), eigs.T)
        tmp = np.dot(np.dot(np.dot(painv, painv.T), YbRinv), ominusf)
        return np.sqrt(nanals - 1) * painv + tmp[:, np.newaxis]
    for k in range(nlevs):
        mask = np.logical_and(covlocal > 1.e-10, oberrs < 1.e10)
        Rinv = np.diag(covlocal[mask] / oberrs[mask])
        ominusf = (obs-hxmean)[mask]
        wts = calcwts(hxprime[:, mask], Rinv, ominusf)
        xens[:, k] = xmean[k] + np.dot(wts.T, xprime[:, k])
    return xens

def letkf_update(xens,hxens,obs,oberrs,covlocal,n_jobs,nlevs_update=None):
    """letkf method"""
    ndim = xens.shape[-1]
    if not n_jobs:
        for n in range(ndim):
            xens[:,:,n] = letkf_kernel(xens[:,:,n],hxens,obs,oberrs,covlocal[:,n],nlevs_update=nlevs_update)
    else:
        # use joblib to distribute over n_jobs tasks
        results = Parallel(n_jobs=n_jobs)(delayed(letkf_kernel)(xens[:,:,n],hxens,obs,oberrs,covlocal[:,n],nlevs_update=nlevs_update) for n in range(ndim))
        for n in range(ndim):
             xens[:,:,n] = results[n]
    return xens

def letkfwts_kernel(hxens,obs,oberrs,covlocal):
    nanals, nobs = hxens.shape
    hxmean = hxens.mean(axis=0)
    hxprime = hxens - hxmean
    wts = np.empty((nanals,nanals),hxens.dtype)
    def calcwts(hx, Rinv, ominusf):
        YbRinv = np.dot(hx, Rinv)
        pa = (nanals - 1) * np.eye(nanals) + np.dot(YbRinv, hx.T)
        evals, eigs = eigh(pa)
        painv = np.dot(np.dot(eigs, np.diag(np.sqrt(1.0 / evals))), eigs.T)
        tmp = np.dot(np.dot(np.dot(painv, painv.T), YbRinv), ominusf)
        return np.sqrt(nanals - 1) * painv + tmp[:, np.newaxis]
    mask = np.logical_and(covlocal > 1.e-10, oberrs < 1.e10)
    Rinv = np.diag(covlocal[mask] / oberrs[mask])
    ominusf = (obs-hxmean)[mask]
    wts = calcwts(hxprime[:, mask], Rinv, ominusf)
    return wts

def letkfwts_kernel2(hxens,obs,oberrs,covlocal):
    nanals, nobs = hxens.shape
    hxmean = hxens.mean(axis=0)
    hxprime = hxens - hxmean
    wts = np.empty((nanals,nanals),hxens.dtype)
    def calcwts(hx, Rinv, ominusf):
        YbRinv = np.dot(hx, Rinv)
        pa = (nanals - 1) * np.eye(nanals) + np.dot(YbRinv, hx.T)
        evals, eigs = eigh(pa)
        painv = np.dot(np.dot(eigs, np.diag(np.sqrt(1.0 / evals))), eigs.T)
        tmp = np.dot(np.dot(np.dot(painv, painv.T), YbRinv), ominusf)
        return np.sqrt(nanals - 1) * painv, tmp
    mask = np.logical_and(covlocal > 1.e-10, oberrs < 1.e10)
    Rinv = np.diag(covlocal[mask] / oberrs[mask])
    ominusf = (obs-hxmean)[mask]
    return calcwts(hxprime[:, mask], Rinv, ominusf)

def letkfwts_compute(hxens,obs,oberrs,covlocal,n_jobs):
    ndim = covlocal.shape[-1]  
    nanals,nobs = hxens.shape
    wts = np.empty((ndim,nanals,nanals),hxens.dtype)
    if not n_jobs:
        for n in range(ndim):
            wts[n] = letkfwts_kernel(hxens,obs,oberrs,covlocal[:,n])
    else:
        # use joblib to distribute over n_jobs tasks
        results = Parallel(n_jobs=n_jobs)(delayed(letkfwts_kernel)(hxens,obs,oberrs,covlocal[:,n]) for n in range(ndim))
        for n in range(ndim):
             wts[n] = results[n]
    return wts

def letkfwts_compute2(hxens,obs,oberrs,covlocal,n_jobs):
    ndim = covlocal.shape[-1]  
    nanals,nobs = hxens.shape
    wts = np.empty((ndim,nanals,nanals),hxens.dtype)
    wtsmean = np.empty((ndim,nanals),hxens.dtype)
    if not n_jobs:
        for n in range(ndim):
            wts[n],wtsmean[n] = letkfwts_kernel2(hxens,obs,oberrs,covlocal[:,n])
    else:
        # use joblib to distribute over n_jobs tasks
        results = Parallel(n_jobs=n_jobs)(delayed(letkfwts_kernel2)(hxens,obs,oberrs,covlocal[:,n]) for n in range(ndim))
        for n in range(ndim):
             wts[n],wtsmean[n] = results[n]
    return wts,wtsmean

def serial_update(xens, hxens, obs, oberrs, covlocal, obcovlocal):
    """serial potter method"""
    nanals, nlevs, ndim = xens.shape
    nobs = len(obs)
    xmean = xens.mean(axis=0)
    xprime = xens - xmean
    hxmean = hxens.mean(axis=0)
    hxprime = hxens - hxmean
    for nob, ob, oberr in zip(np.arange(nobs), obs, oberrs):
        if oberr > 1.e10: continue
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
