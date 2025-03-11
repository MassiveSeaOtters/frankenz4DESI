#!/usr/bin/env python

from functools import partial
import numpy as np
from typing import Callable, Tuple, Optional, Union
from yacs.config import CfgNode as CN
from .defs import *


def identity(flux: np.ndarray, err: np.ndarray)->Tuple[np.ndarray, np.ndarray]:
    """
    Identity mapping
    ---------------------
    Args:
        flux (np.ndarray, uJy): observed photometric flux densities
        err (np.ndarray, uJy): observed photometric flux density errors

    Returns:
        flux (np.ndarray, uJy): observed photometric flux densities
        err (np.ndarray, uJy): observed photometric flux density errors
    """
    return flux, err


def magnitude(flux: np.ndarray, err: np.ndarray, zeropoints: Optional[Union[np.ndarray, float]]=ABzpt)->Tuple[np.ndarray, np.ndarray]:
    """
    Convert photometry to magnitude
    -----------------------
    Args:
        flux (np.ndarray, uJy): observed photometric flux densities
        err (np.ndarray, uJy): observed photometric flux density errors
        zeropoints (np.ndarray|float, uJy): zero point of the magnitude systems, default AB magnitude

    Returns:
        magnitude (np.ndarray): AB magnitudes corresponding to flux
        magnitude error (np.ndarray): AB magnitude errors corresponding to err
    """
    mag = -2.5*np.log10(flux/zeropoints)
    mag_err = np.abs(err/flux)*2.5/np.log(10)
    return mag, mag_err


def luptitude(flux: np.ndarray, err: np.ndarray, skynoise: Optional[Union[np.ndarray, float]]=fdepths, zeropoints: Optional[Union[np.ndarray, float]]=ABzpt):
    """
    Convert photometry to asinh magnitudes (i.e. "Luptitudes"). 
    See Lupton et al. (1999) for more details.
    https://ned.ipac.caltech.edu/help/sdss/dr6/photometry.html
    ----------------------------------------------------------
    Args:
        flux (np.ndarray, uJy): observed photometric flux densities
        err (np.ndarray, uJy): observed photometric flux density errors
        skynoise (np.ndarray|float, uJy): background sky noise, used as a "softening parameter", default HSC 1-sigma depth
        zeropoints (np.ndarray|float, uJy): flux density zero-points, used as a "location parameter", default zero-point of AB magnitude

    Returns
    -------
    magnitude (np.ndarray): asinh magnitudes corresponding to input phot
    magnitude error (np.ndarray): asinh magnitudes errors corresponding to input err
    """

    mag = -2.5 / np.log(10.) * (np.arcsinh(flux / (2. * skynoise)) + np.log(skynoise / zeropoints))
    mag_err = np.sqrt(np.square(2.5 * np.log10(np.e) * err) / (np.square(2. * skynoise) + np.square(flux)))

    return mag, mag_err


def get_transfrom(config: CN)->Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    get transform function from config
    """
    if config.TRANSFORM.TYPE == 'identity':
        return identity
    if config.TRANSFORM.TYPE == 'magnitude':
        return partial(magnitude, zeropoints=np.array(config.TRANSFORM.ZPT))
    if config.TRANSFORM.TYPE == 'luptitude':
        return partial(luptitude, skynoise=np.array(config.TRANSFORM.SKYNOISE), zeropoints=np.array(config.TRANSFORM.ZPT))
    else:
        raise NotImplementedError(f"{config.TRANSFORM.TYPE} is not implemented...")
