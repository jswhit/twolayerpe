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


def enkf_update(
    xens, hxens, obs, oberrs, covlocal, obcovlocal=None, denkf=False):
    """serial potter method or LETKF (if obcovlocal is None)"""

    # no vertical localization!

    nanals, nlevs, ndim = xens.shape
    nobs = len(obs)
    xmean = xens.mean(axis=0)
    xprime = xens - xmean
    hxmean = hxens.mean(axis=0)
    hxprime = hxens - hxmean
    eps = 1.e-7

    if obcovlocal is not None:  # serial EnSRF update

        for nob, ob, oberr in zip(np.arange(nobs), obs, oberrs):
            ominusf = ob - hxmean[nob].copy()
            hxens = hxprime[:, nob].copy().reshape((nanals, 1))
            hpbht = (hxens ** 2).sum() / (nanals - 1)
            if denkf:
                gainfact = 0.5
            else:
                gainfact = (
                    (hpbht + oberr)
                    / hpbht
                    * (1.0 - np.sqrt(oberr / (hpbht + oberr)))
                )
            # state space update
            # only update points closer than localization radius to ob
            mask = covlocal[nob, :] > eps
            for k in range(nlevs):
                pbht = (xprime[:, k, mask].T * hxens[:, 0]).sum(axis=1) / float(
                    nanals - 1
                )
                kfgain = covlocal[nob, mask] * pbht / (hpbht + oberr)
                xmean[k, mask] = xmean[k, mask] + kfgain * ominusf
                xprime[:, k, mask] = xprime[:, k, mask] - gainfact * kfgain * hxens
            # observation space update
            # only update obs within localization radius
            mask = obcovlocal[nob, :] > eps
            pbht = (hxprime[:, mask].T * hxens[:, 0]).sum(axis=1) / float(
                nanals - 1
            )
            kfgain = obcovlocal[nob, mask] * pbht / (hpbht + oberr)
            hxmean[mask] = hxmean[mask] + kfgain * ominusf
            hxprime[:, mask] = (
                hxprime[:, mask] - gainfact * kfgain * hxens
            )
        return xmean + xprime

    else:  # LETKF update

        def calcwts(hx, Rinv, ominusf):
            YbRinv = np.dot(hx, Rinv)
            pa = (nanals - 1) * np.eye(nanals) + np.dot(YbRinv, hx.T)
            if denkf:  # just return what's needed to compute Kalman Gain
                return np.dot(cho_solve(cho_factor(pa), np.eye(nanals)), YbRinv)
            evals, eigs = np.linalg.eigh(pa)
            painv = np.dot(np.dot(eigs, np.diag(np.sqrt(1.0 / evals))), eigs.T)
            tmp = np.dot(np.dot(np.dot(painv, painv.T), YbRinv), ominusf)
            return np.sqrt(nanals - 1) * painv + tmp[:, np.newaxis]

        omf = obs - hxmean
        for n in range(ndim): # horizontal grid
            for k in range(nlevs):
                mask = covlocal[:,n] > eps
                Rinv = np.diag(covlocal[mask, n] / oberrs[mask])
                ominusf = omf[mask]
                wts = calcwts(hxprime[:, mask], Rinv, ominusf)
                if denkf:
                    kfgain = np.dot(wts.T, xprime[:, k, n])
                    xmean[k, n] += np.dot(kfgain, ominusf)
                    xprime[:, k, n] -= 0.5 * np.dot(kfgain, hxprime[:, mask].T)
                    xens[:, k, n] = xmean[k, n] + xprime[:, k, n]
                else:
                    xens[:, k, n] = xmean[k, n] + np.dot(wts.T, xprime[:, k, n])
        return xens
