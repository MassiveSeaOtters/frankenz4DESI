#!/usr/bin/env python

import os
import sys
import h5py
import warnings
import numpy as np
from typing import Tuple, Optional
from functools import partial
from frankenz.pdf import loglike, PDFDict, gauss_kde_dict, gauss_kde
from yacs.config import CfgNode as CN
from scipy.special import logsumexp
from tqdm import tqdm

from .utils import load_hdf5, construct_KDTree
from .transforms import get_transfrom
from .priors import get_prior


class frankenz4DESIKNN(object):
    def __init__(self, config: CN) -> None:
        self.transform = get_transfrom(config)
        self._model = load_hdf5(config.DATA.MODEL, name=config.DATA.NAME, transform=self.transform, verbose=config.VERBOSE)
        self.KDTreeList = construct_KDTree(self._model['flux'], self._model['flux_err'], 
                                        self.transform, config.MODEL.KTREE, config.VERBOSE, 
                                        leafsize=config.MODEL.KDTREE.LEAFSIZE, compact_nodes=config.MODEL.KDTREE.COMPACT_NODES,
                                        copy_data=config.MODEL.KDTREE.COPY_DATA, balanced_tree=config.MODEL.KDTREE.BALANCED_TREE,
                                        boxsize=config.MODEL.KDTREE.BOXSIZE)
        
        self.model_mask = np.atleast_2d(np.ones_like(self._model['flux'], dtype=bool))
        
        self.K = config.MODEL.KTREE
        self.k = config.MODEL.KPOINT
        self.Nmodels = self.K * self.k
        self.eps = config.MODEL.KDTREE.EPS
        self.lp_norm = config.MODEL.KDTREE.LPNORM
        self.distance_upper_bound = config.MODEL.KDTREE.DISTANCE_UPPER_BOUND
        
        self.prior = get_prior(config)
        self.gauss_kde = partial(gauss_kde_dict, wt_thresh=config.MODEL.PDF.WT_THRESH, cdf_thresh=config.MODEL.PDF.CDF_THRESH)

        self.verbose = config.VERBOSE

        self.zgrid = np.arange(config.MODEL.ZGRID.ZSTART,
                               config.MODEL.ZGRID.ZEND,
                               config.MODEL.ZGRID.ZDELTA)
        self.zsmooth = config.MODEL.ZSMOOTH
        self.enable_zerr = config.MODEL.ENABLE_ZERR
        self.wt_thresh = config.MODEL.PDF.WT_THRESH
        self.cdf_thresh = config.MODEL.PDF.CDF_THRESH


    
    def loglike_for_single_data(self, d: np.ndarray, e: np.ndarray, m: np.ndarray)->Tuple:
        d, e, m = list(map(np.atleast_2d, [d, e, m]))
        d_t = np.random.normal(d.repeat(self.K, axis=0), e.repeat(self.K, axis=0))
        f_t, _ = self.transform(d_t, e.repeat(self.K, axis=0))
        indices = np.array([T.query(f, 
                                    k=self.k, 
                                    eps=self.eps, 
                                    p=self.lp_norm, 
                                    distance_upper_bound=self.distance_upper_bound)[1][0]
                                    for f, T in zip(f_t, self.KDTreeList)]).flatten()
        idxs = np.unique(indices)
        Nidx = len(idxs)
        ll, Ndim, chi2 = loglike(d, e, m, self.model_flux, self.model_err, self.model_mask)
        return ll[idxs], Ndim[idxs], chi2[idxs], idxs, Nidx


    def loglike(self, data: np.ndarray, data_err: np.ndarray, data_mask: np.ndarray):
        Ndata = len(data)
        Nneighbors = np.zeros(Ndata, dtype=int)
        neighbors = np.zeros((Ndata, self.Nmodels), dtype=int) - 99
        fit_lnlike = np.zeros((Ndata, self.Nmodels), dtype=float) - np.inf
        fit_Ndim = np.zeros((Ndata, self.Nmodels), dtype=int)
        fit_chi2 = np.zeros((Ndata, self.Nmodels), dtype=float) + np.inf

        for idx in tqdm(range(Ndata), disable=(not self.verbose), 
                        desc='computing log-likelihood'):
            ll, Ndim, chi2, idxs, Nidx = self.loglike_for_single_data(data[idx], data_err[idx], data_mask[idx])
            Nneighbors[idx] = Nidx
            neighbors[idx, :Nidx] = idxs
            fit_lnlike[idx, :Nidx] = ll
            fit_Ndim[idx, :Nidx] = Ndim
            fit_chi2[idx, :Nidx] = chi2
        return Nneighbors, neighbors, fit_lnlike, fit_Ndim, fit_chi2
    

    def logprob(self, data: np.ndarray, data_err: np.ndarray, data_mask: np.ndarray):
        Ndata = len(data)
        model_lnpriors_num = self.prior(self.model_flux, self.model_err, self.model_mask, data, data_err, data_mask, self.transform)
        model_lnpriors_den = self.prior(self.model_flux, self.model_err, self.model_mask, data, data_err, data_mask, self.transform)
        model_lnpriors_ratio = model_lnpriors_num - model_lnpriors_den
        Nneighbors, neighbors, fit_lnlike, fit_Ndim, fit_chi2 = self.loglike(data, data_err, data_mask)
        fit_lnprior_ratio = np.zeros((Ndata, self.Nmodels), dtype=np.float32) - np.inf
        fit_lnprob = np.zeros((Ndata, self.Nmodels), dtype=np.float32) - np.inf
        fit_wt = np.zeros((Ndata, self.Nmodels), dtype=np.float32) - np.inf
        fit_lmap = np.zeros((Ndata, ), dtype=np.float32) - np.inf
        fit_levid = np.zeros((Ndata, self.Nmodels), dtype=np.float32) - np.inf
        for idx in tqdm(range(Ndata), disable=(not self.verbose), desc='computing log-prob'):
            fit_lnprior_ratio[idx, :Nneighbors[idx]] = model_lnpriors_ratio[neighbors[idx, :Nneighbors[idx]]].copy()
            fit_lnprob[idx, :Nneighbors[idx]] = fit_lnlike[idx, :Nneighbors[idx]].copy() + fit_lnprior_ratio[idx, :Nneighbors[idx]].copy()
            lnprob = fit_lnprob[idx, :Nneighbors[idx]].copy()
            lmap, levid = max(lnprob), logsumexp(lnprob)
            fit_lmap[idx] = lmap
            fit_levid[idx, :Nneighbors[idx]] = levid
            wt = np.exp(lnprob - levid)
            fit_wt[idx, :Nneighbors[idx]] = wt
        return Nneighbors, neighbors, fit_lnlike, fit_lnprior_ratio, fit_lnprob, fit_wt, fit_Ndim, fit_chi2, fit_lmap, fit_levid
    

    def predict(self, data: np.ndarray, data_err: np.ndarray, data_mask: np.ndarray,
                output_path: Optional[str]=None, object_id: Optional[np.ndarray]=None,
                z: Optional[np.ndarray]=None):
        Ndata = len(data)
        result = self.logprob(data, data_err, data_mask)
        Nneighbors, neighbors, fit_lnlike, fit_lnprior_ratio, fit_lnprob, fit_wt, fit_Ndim, fit_chi2, fit_lmap, fit_levid = result
        pdflist = []
        for idx in tqdm(range(Ndata), disable=(not self.verbose), desc='computing pdf'):
            idxs = neighbors[idx, :Nneighbors[idx]]
            pdf = gauss_kde(self.model_redshift[idxs], 
                            self.model_redshift_err[idxs],
                            self.zgrid, 
                            y_wt=fit_wt[idx, :Nneighbors[idx]],
                            wt_thresh=self.wt_thresh,
                            cdf_thresh=self.cdf_thresh)
            pdf /= pdf.sum()
            pdflist.append(pdf)
        pdflist = np.array(pdflist)

        if not output_path:
            return pdflist
        else:
            file = h5py.File(output_path, mode='w')
            model_grp = file.create_group('model')
            model_grp.create_dataset('object_id', data=self._model['object_id'], dtype=np.int64)
            model_grp.create_dataset('flux', data=self.model_flux, dtype=np.float32)
            model_grp.create_dataset('err', data=self.model_err, dtype=np.float32)
            model_grp.create_dataset('z', data=self.model_redshift, dtype=np.float32)
            model_grp.create_dataset('zerr', data=self.model_redshift_err, dtype=np.float32)
            result_grp = file.create_group('result')
            if object_id is not None:
                result_grp.create_dataset('object_id', data=object_id, dtype=np.int64)
            if z is not None:
                result_grp.create_dataset('z', data=z, dtype=np.float32)
            result_grp.create_dataset('pdf', data=pdflist, dtype=np.float32)
            result_grp.create_dataset('Nneighbors', data=Nneighbors, dtype=int)
            result_grp.create_dataset('neighbors', data=neighbors, dtype=int)
            result_grp.create_dataset('fit_lnlike', data=fit_lnlike, dtype=np.float32)
            result_grp.create_dataset('fit_lnprior_ratio', data=fit_lnprior_ratio, dtype=np.float32)
            result_grp.create_dataset('fit_lnprob', data=fit_lnprob, dtype=np.float32)
            result_grp.create_dataset('fit_wt', data=fit_wt, dtype=np.float32)
            result_grp.create_dataset('fit_Ndim', data=fit_Ndim, dtype=int)
            result_grp.create_dataset('fit_chi2', data=fit_chi2, dtype=np.float32)
            result_grp.create_dataset('fit_lmap', data=fit_lmap, dtype=np.float32)
            result_grp.create_dataset('fit_levid', data=fit_levid, dtype=np.float32)
            file.close()
            return pdflist


    @property
    def model_object_id(self):
        return self._model['object_id']

    @property
    def model_flux(self):
        return self._model['flux']

    @property
    def model_err(self):
        return self._model['flux_err']
    
    @property
    def model_redshift(self):
        return self._model['z']
    
    @property
    def model_redshift_err(self):
        return (self._model['zerr'] if self.enable_zerr else 0.) + np.ones_like(self._model['zerr'], dtype=np.float32)