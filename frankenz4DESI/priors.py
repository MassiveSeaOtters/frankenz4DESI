#!/usr/bin/env python

import sys
import numpy as np
from scipy.special import logsumexp
from frankenz.pdf import loglike
from tqdm import tqdm
from functools import partial
from yacs.config import CfgNode as CN

from .utils import construct_KDTree




def _prior_knn(model, model_err, model_mask, data, data_err, data_mask, transform,
               K=25, k=20, verbose=True, leafsize=10, compact_nodes=True, copy_data=False, balanced_tree=True, 
               boxsize=None, eps=1e-3, p=2, distance_upper_bound=np.inf):
    if verbose:
        sys.stderr.write('compute knn prior...\n')
        sys.stderr.flush()
    KDTreeList = construct_KDTree(data, data_err, transform, K, verbose, leafsize=leafsize, 
                                  compact_nodes=compact_nodes, balanced_tree=balanced_tree,
                                  copy_data=copy_data, boxsize=boxsize)
    Nmodel = len(model)
    model_lnprior = np.zeros(Nmodel, dtype=np.float32)
    for idx in tqdm(range(Nmodel), disable=(not verbose), desc='computing knn prior'):
        m, me, mm = list(map(np.atleast_2d, [model[idx], model_err[idx], model_mask[idx]]))
        m_t = np.random.normal(m.repeat(K, axis=0), me.repeat(K, axis=0))
        f_t, _ = transform(m_t, me.repeat(K, axis=0))
        indices = np.array([T.query(f, 
                                    k=k, 
                                    eps=eps, 
                                    p=p, 
                                    distance_upper_bound=distance_upper_bound)[1][0]
                                    for f, T in zip(f_t, KDTreeList)]).flatten()
        idxs = np.unique(indices)
        Nidx = len(idxs)
        ll, _, _ = loglike(m, me, mm, data, data_err, data_mask)
        model_lnprior[idx] = logsumexp(ll[idxs])
    return model_lnprior


def _prior_uniform(model, model_err, model_mask, data, data_err, data_mask, transform):
    return np.ones(model.shape[0], dtype=float)


def get_prior(config: CN):
    if config.MODEL.PRIOR.TYPE == 'uniform':
        return _prior_uniform
    if config.MODEL.PRIOR.TYPE == 'knn':
        return partial(_prior_knn, 
                       K=config.MODEL.PRIOR.KTREE, 
                       k=config.MODEL.PRIOR.KPOINT,
                       verbose=config.VERBOSE,
                       leafsize=config.MODEL.PRIOR.LEAFSIZE,
                       compact_nodes=config.MODEL.PRIOR.COMPACT_NODES,
                       copy_data=config.MODEL.PRIOR.COPY_DATA,
                       balanced_tree=config.MODEL.PRIOR.BALANCED_TREE,
                       boxsize=config.MODEL.PRIOR.BOXSIZE,
                       eps=config.MODEL.PRIOR.EPS,
                       p=config.MODEL.PRIOR.LPNORM,
                       distance_upper_bound=config.MODEL.PRIOR.DISTANCE_UPPER_BOUND)
    else:
        raise NotImplementedError(f"{config.MODEL.PRIOR.TYPE} hasn't been implemented!")
    
    



