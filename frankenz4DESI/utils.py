#!/usr/bin/env python

import h5py
import sys
import numpy as np
from astropy import units as u
from scipy.spatial import KDTree
from typing import Optional, Callable, Tuple, Dict, List


def load_hdf5(path: str, name: str, transform: Optional[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]=None, verbose: Optional[bool]=True)->Dict[str, np.ndarray]:
    """
    path (str): file path of the hdf5 file
    name (str): which kind of photometry to be used
    transform (function): transform input flux and err to desired features, if None, return flux and err
    verbose (bool): whether or not to print informations
    """
    file = h5py.File(path, 'r')
    assert name in file.keys(), f"{name} doesn't in this hdf5 file"
    raw_a = np.array(file['a'])
    raw_flux, raw_err, flag = np.array(file[f'{name}/flux']), np.array(file[f'{name}/err']), np.array(file[f'{name}/flag'])
    flag = (flag.sum(axis=1)==0)
    masked_flux, masked_err, masked_a = raw_flux[flag], raw_err[flag], raw_a[flag]
    masked_mag = (masked_flux*u.uJy).to(u.ABmag).value - masked_a
    corr_flux = ((masked_mag*u.ABmag).to(u.uJy)).value
    if transform:
        feature, feature_err = transform(corr_flux, masked_err)
    else:
        feature, feature_err = corr_flux, masked_err
    
    masked_object_id = np.array(file['object_id'])[flag] if 'object_id' in file.keys() else None
    masked_z = np.array(file['z'])[flag] if 'z' in file.keys() else None
    masked_zerr = np.array(file['zerr'])[flag] if 'zerr' in file.keys() else None
    
    data  = {
        'object_id': masked_object_id,
        'z': masked_z,
        'zerr': masked_zerr,
        'flux': masked_flux,
        'flux_err': masked_err,
        'feature': feature,
        'feature_err': feature_err
    }
    file.close()
    if verbose:
        sys.stderr.write(f'successfully loaded {np.sum(flag)} data from {flag.shape[0]} data, success rate: {np.sum(flag)/flag.shape[0]*100}%\n')
        sys.stderr.flush()
    return data


def construct_KDTree(flux: np.ndarray, err: np.ndarray, transform: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]], K: int, verbose: bool, **kdkwargs)->List[KDTree]:
    TreeList = []
    for i in range(K):
        flux_sample = np.array(np.random.normal(flux, err), dtype=np.float32)
        feature_sample, _ = transform(flux_sample, err)
        TreeList.append(KDTree(feature_sample, **kdkwargs))
        if verbose:
            sys.stderr.write('\r{}/{} KDTrees constructed'.format(i+1, K))
            sys.stderr.flush()
    if verbose:
        sys.stderr.write('\n')
        sys.stderr.flush()
    return TreeList



def pdfs_summarize(pdfs, pgrid, renormalize=True, rstate=None,
                   pkern='lorentz', pkern_grid=None, wconf_func=None):
    """
    Compute PDF summary statistics. Point estimators include:

    * mean: optimal estimator under L2 loss
    * median: optimal estimator under L1 loss
    * mode: optimal estimator under L0 (pseudo-)loss
    * best: optimal estimator under loss from `pkern` and `pkern_grid`

    Estimators also come with multiple quality metrics attached:

    * std: standard deviation computed around the estimator
    * conf: fraction of the PDF contained within a window around the estimator
    * risk: associated risk computed under the loss from `pkern`

    68% and 95% lower/upper credible intervals are also reported.

    For statistical purposes, a Monte Carlo realization of the posterior
    is also generated.

    Based on code from Sogo Mineo and Atsushi Nishizawa and used in the
    HSC-SSP DR1 photo-z release.

    Parameters
    ----------
    pdfs : `~numpy.ndarray` with shape (Npdf, Ngrid)
        Original collection of PDFs.

    pgrid : `~numpy.ndarray` with shape (Ngrid)
        Grid the PDFs are evaluated over.

    renormalize : bool, optional
        Whether to renormalize the PDFs before computation. Default is `True`.

    rstate : `~numpy.random.RandomState` instance, optional
        Random state instance. If not passed, the default `~numpy.random`
        instance will be used.

    pkern : str or func, optional
        The kernel used to compute the effective loss over the grid when
        computing the `best` estimator. Default is `'lorentz'`.

    pkern_grid : `~numpy.ndarray` with shape (Ngrid, Ngrid), optional
        The 2-D array of positions that `pkern` is evaluated over.
        If not provided, a `1. / ((1. + x) * sig)` weighting over `pgrid`
        will be used, where `sig = 0.15`. **Note that this is designed for
        photo-z estimation and will not be suitable for most problems.**

    wconf_func : func, optional
        A function that takes an input point and generates an associated
        +/- width value. Used to construct `conf` estimates.

    Returns
    -------
    (pmean, pmean_std, pmean_conf, pmean_risk) : 4-tuple with `~numpy.ndarray`
    elements of shape (Nobj)
        Mean estimator and associated uncertainty/quality assessments.

    (pmed, pmed_std, pmed_conf, pmed_risk) : 4-tuple with `~numpy.ndarray`
    elements of shape (Nobj)
        Median estimator and associated uncertainty/quality assessments.

    (pmode, pmode_std, pmode_conf, pmode_risk) : 4-tuple with `~numpy.ndarray`
    elements of shape (Nobj)
        Mode estimator and associated uncertainty/quality assessments.

    (pbest, pbest_std, pbest_conf, pbest_risk) : 4-tuple with `~numpy.ndarray`
    elements of shape (Nobj)
        "Best" estimator and associated uncertainty/quality assessments.

    (plow95, plow68, phigh68, phigh95) : 4-tuple with `~numpy.ndarray` elements
    of shape (Nobj)
        Lower 95%, lower 68%, upper 68%, and upper 95% quantiles.

    pmc : `~numpy.ndarray` of shape (Nobj)
        Monte Carlo realization of the posterior.

    """

    if rstate is None:
        rstate = np.random

    Nobj, Ngrid = len(pdfs), len(pgrid)
    if renormalize:
        pdfs /= pdfs.sum(axis=1)[:, None]  # sum to 1

    # Compute mean.
    pmean = np.dot(pdfs, pgrid)

    # Compute mode.
    pmode = pgrid[np.argmax(pdfs, axis=1)]

    # Compute CDF-based quantities.
    cdfs = pdfs.cumsum(axis=1)
    plow2, phigh2 = np.zeros(Nobj), np.zeros(Nobj)  # +/- 95%
    plow1, phigh1 = np.zeros(Nobj), np.zeros(Nobj)  # +/- 68%
    pmed = np.zeros(Nobj)  # median
    pmc = np.zeros(Nobj)  # Monte Carlo realization
    for i, cdf in enumerate(cdfs):
        qs = [0.025, 0.16, 0.5, 0.84, 0.975, rstate.rand()]  # quantiles
        qvals = np.interp(qs, cdf, pgrid)
        plow2[i], plow1[i], pmed[i], phigh1[i], phigh2[i], pmc[i] = qvals

    # Compute kernel-based quantities.
    if pkern_grid is None:
        # Structure grid of "truth" values and "guess" values.
        # **Designed for photo-z estimation -- likely not applicable in most
        # other applications.**
        ptrue = pgrid.reshape(Ngrid, 1)
        pguess = pgrid.reshape(1, Ngrid)
        psig = 0.15  # kernel dispersion
        pkern_grid = (ptrue - pguess) / ((1. + ptrue) * 0.15)
    if pkern == 'tophat':
        # Use top-hat kernel
        kernel = (np.square(pkern_grid) < 1.)
    elif pkern == 'gaussian':
        kernel = np.exp(-0.5 * np.square(pkern_grid))
    elif pkern == 'lorentz':
        kernel = 1. / (1. + np.square(pkern_grid))
    else:
        try:
            kernel = pkern(pkern_grid)
        except:
            raise RuntimeError("The input kernel does not appear to be valid.")
    prisk = np.dot(pdfs, 1.0 - kernel)  # "risk" estimator
    pbest = pgrid[np.argmin(prisk, axis=1)]  # "best" estimator

    # Compute second moment uncertainty estimate (i.e. std-dev).
    grid = pgrid.reshape(1, Ngrid)
    sqdev = np.square(grid - pmean.reshape(Nobj, 1))  # mean
    pmean_std = np.sqrt(np.sum(sqdev * pdfs, axis=1))
    sqdev = np.square(grid - pmed.reshape(Nobj, 1))  # med
    pmed_std = np.sqrt(np.sum(sqdev * pdfs, axis=1))
    sqdev = np.square(grid - pmode.reshape(Nobj, 1))  # mode
    pmode_std = np.sqrt(np.sum(sqdev * pdfs, axis=1))
    sqdev = np.square(grid - pbest.reshape(Nobj, 1))  # best
    pbest_std = np.sqrt(np.sum(sqdev * pdfs, axis=1))

    # Construct "confidence" estimates around our primary point estimators
    # (i.e. how much of the PDF is contained within +/= some fixed interval).
    if wconf_func is None:
        def wconf_func(point):
            return (1. + point) * 0.03
    pmean_conf, pmed_conf, pmode_conf, pbest_conf = np.zeros((4, Nobj))
    for i, cdf in enumerate(cdfs):
        # Mean
        width = wconf_func(pmean[i])
        pmean_low, pmean_high = pmean[i] - width, pmean[i] + width
        # Median
        width = wconf_func(pmed[i])
        pmed_low, pmed_high = pmed[i] - width, pmed[i] + width
        # Mode
        width = wconf_func(pmode[i])
        pmode_low, pmode_high = pmode[i] - width, pmode[i] + width
        # "Best"
        width = wconf_func(pbest[i])
        pbest_low, pbest_high = pbest[i] - width, pbest[i] + width
        # Interpolate CDFs.
        qs = np.array([pmean_low, pmean_high, pmed_low, pmed_high, pmode_low,
                       pmode_high, pbest_low, pbest_high])
        qvs = np.interp(qs, pgrid, cdf)
        (pmean_conf[i], pmed_conf[i],
         pmode_conf[i], pbest_conf[i]) = qvs[[1, 3, 5, 7]] - qvs[[0, 2, 4, 6]]

    # Construct "risk" estimates around our primary point estimators.
    pmean_risk, pmed_risk, pmode_risk, pbest_risk = np.zeros((4, Nobj))
    for i, pr in enumerate(prisk):
        vals = np.interp([pmean[i], pmed[i], pmode[i], pbest[i]], pgrid, pr)
        pmean_risk[i], pmed_risk[i], pmode_risk[i], pbest_risk[i] = vals

    return ((pmean, pmean_std, pmean_conf, pmean_risk),
            (pmed, pmed_std, pmed_conf, pmed_risk),
            (pmode, pmode_std, pmode_conf, pmode_risk),
            (pbest, pbest_std, pbest_conf, pbest_risk),
            (plow2, plow1, phigh1, phigh2), pmc)




        

