import numpy as np
#import time
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

def modens(enspert,sqrtcovlocal):
    nanals = enspert.shape[0]
    neig = sqrtcovlocal.shape[0]
    #nlevs = enspert.shape[1]
    #Nt = enspert.shape[2]
    #enspertm = np.empty((nanals*neig,nlevs,Nt,Nt),enspert.dtype)
    #for k in range(nlevs):
    #   nanal2 = 0
    #   for j in range(neig):
    #       for nanal in range(nanals):
    #           enspertm[nanal2,k,...] = enspert[nanal,k,...]*sqrtcovlocal[j,...]
    #           nanal2+=1
    enspertm = np.multiply(np.repeat(sqrtcovlocal[:,np.newaxis,:,:],nanals,axis=0),np.tile(enspert,(neig,1,1,1)))
    #print(sqrtcovlocal.min(), sqrtcovlocal.max(),(enspertm-enspertm2).min(), (enspertm-enspertm2).max())
    return enspertm

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
        evals, eigs = eigh(pa,driver='evd')
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
        evals, eigs = eigh(pa,driver='evd')
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
        evals, eigs = eigh(pa,driver='evd')
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

def getkf(hxprime_b,hxprime_bm,oberrvar,dep,denkf=False):
    oberrstd = np.sqrt(oberrvar[np.newaxis,:])
    nanals = hxprime_b.shape[0] # 'original' ensemble size
    nanals2 = hxprime_bm.shape[0] # modulated ensemble size
    # getkf global solution with model-space localization
    # HZ^T = hxens * R**-1/2
    # compute eigenvectors/eigenvalues of HZ^T HZ (C=left SV)
    # (in Bishop paper HZ is nobs, nanals, here is it nanals, nobs)
    # normalize so dot product is covariance
    normfact = np.array(np.sqrt(nanals-1),dtype=np.float32)
    hxtmp = (hxprime_bm/oberrstd)/normfact
    pa = np.dot(hxtmp,hxtmp.T)
    #t1 = time.time()
    evals, evecs = eigh(pa, driver='evd')
    #t2 = time.time()
    #print('time in eigh',t2-t1)
    gamma_inv = np.zeros_like(evals)
    #evals = np.where(evals > np.finfo(evals.dtype).eps, evals, 0.)
    #gamma_inv = np.where(evals > np.finfo(evals.dtype).eps, 1./evals, 0.)
    for n in range(nanals2):
        if evals[n] > np.finfo(evals.dtype).eps:
           gamma_inv[n] = 1./evals[n]
        else:
           evals[n] = 0. 
    # gammapI used in calculation of posterior cov in ensemble space
    gammapI = evals+1.
    # create HZ^T R**-1/2 
    shxtmp = hxtmp/oberrstd
    # compute factor to multiply with model space ensemble perturbations
    # to compute analysis increment (for mean update), save in single precision.
    # This is the factor C (Gamma + I)**-1 C^T (HZ)^ T R**-1/2 (y - HXmean)
    # in Bishop paper (eqs 10-12).
    # pa = C (Gamma + I)**-1 C^T (analysis error cov in ensemble space)
    #    = matmul(evecs/gammapI,transpose(evecs))
    pa = np.dot(evecs/gammapI[np.newaxis,:],evecs.T)
    # wts_ensmean = C (Gamma + I)**-1 C^T (HZ)^ T R**-1/2 (y - HXmean)
    # (nanals, nanals) x (nanals,) = (nanals,)
    # do nanal=1,nanals
    #    swork1(nanal) = sum(shxtmp(nanal,:)*dep(:))
    # end do
    # do nanal=1,nanals
    #    wts_ensmean(nanal) = sum(pa(nanal,:)*swork1(:))/normfact
    # end do
    wts_ensmean = np.dot(pa, np.dot(shxtmp,dep))/normfact
    # compute factor to multiply with model space ensemble perturbations
    # to compute analysis increment (for perturbation update), save in single precision.
    # This is -C [ (I - (Gamma+I)**-1/2)*Gamma**-1 ] C^T (HZ)^T R**-1/2 HXprime
    # in Bishop paper (eqn 29).
    # For DEnKF factor is -0.5*C (Gamma + I)**-1 C^T (HZ)^ T R**-1/2 HXprime
    # = -0.5 Pa (HZ)^ T R**-1/2 HXprime (Pa already computed)
    # pa = C [ (I - (Gamma+I)**-1/2)*Gamma**-1 ] C^T
    # gammapI = sqrt(1.0/gammapI)
    # do nanal=1,nanals
    #    swork3(nanal,:) = &
    #    evecs(nanal,:)*(1.-gammapI(:))*gamma_inv(:)
    # enddo
    # pa = matmul(swork3,transpose(swork2))
    if denkf:
        pa=0.5*pa # for denkf
    else:
        pa=np.dot(evecs*(1.-np.sqrt(1./gammapI[np.newaxis,:]))*gamma_inv[np.newaxis,:],evecs.T)
    # wts_ensperts = -C [ (I - (Gamma+I)**-1/2)*Gamma**-1 ] C^T (HZ)^T R**-1/2 HXprime
    # (nanals, nanals) x (nanals, nanals/eigv) = (nanals, nanals/neigv)
    # if denkf, wts_ensperts = -0.5 C (Gamma + I)**-1 C^T (HZ)^T R**-1/2 HXprime
    # swork2 = matmul(shxtmp,transpose(hxens_orig))
    #wts_ensperts = -matmul(pa, swork2)/normfact
    wts_ensperts = -np.dot(pa, np.dot(shxtmp,hxprime_b.T)).T/normfact
    #paens = pa/normfact**2 # posterior covariance in modulated ensemble space
    return wts_ensmean,wts_ensperts
