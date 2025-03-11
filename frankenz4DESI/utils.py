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








        

