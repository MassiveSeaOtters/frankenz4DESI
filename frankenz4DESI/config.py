#!/usr/bin/env python

import os
import sys
import yaml
import numpy as np
from yacs.config import CfgNode as CN
from .defs import *


_C = CN()

_C.BASE = ['']
_C.VERBOSE = True


_C.DATA = CN()
_C.DATA.MODEL = ''
_C.DATA.DATA = ''
_C.DATA.OUTPUT = ''
_C.DATA.NAME = 'cmodel'


_C.MODEL = CN()
_C.MODEL.KTREE = 25
_C.MODEL.KPOINT = 20
_C.MODEL.ZSMOOTH = 0.01
_C.MODEL.ENABLE_ZERR = False


_C.MODEL.KDTREE = CN()
_C.MODEL.KDTREE.LEAFSIZE = 20
_C.MODEL.KDTREE.COMPACT_NODES = True
_C.MODEL.KDTREE.COPY_DATA = False
_C.MODEL.KDTREE.BALANCED_TREE = True
_C.MODEL.KDTREE.BOXSIZE = None

_C.MODEL.KDTREE.EPS = 1e-3
_C.MODEL.KDTREE.LPNORM = 2
_C.MODEL.KDTREE.DISTANCE_UPPER_BOUND = np.inf


_C.MODEL.PRIOR = CN()
_C.MODEL.PRIOR.TYPE = 'uniform'
_C.MODEL.PRIOR.KTREE = 25
_C.MODEL.PRIOR.KPOINT = 20

_C.MODEL.PRIOR.LEAFSIZE = 20
_C.MODEL.PRIOR.COMPACT_NODES = True
_C.MODEL.PRIOR.COPY_DATA = False
_C.MODEL.PRIOR.BALANCED_TREE = True
_C.MODEL.PRIOR.BOXSIZE = None
_C.MODEL.PRIOR.EPS = 1e-3
_C.MODEL.PRIOR.LPNORM = 2
_C.MODEL.PRIOR.DISTANCE_UPPER_BOUND = np.inf


_C.MODEL.ZGRID = CN()
_C.MODEL.ZGRID.ZSTART = 0
_C.MODEL.ZGRID.ZEND = 7.
_C.MODEL.ZGRID.ZDELTA = 1e-2


_C.MODEL.PDF = CN()
_C.MODEL.PDF.WT_THRESH = 1e-3
_C.MODEL.PDF.CDF_THRESH = 2e-4


_C.TRANSFORM = CN()
_C.TRANSFORM.TYPE = 'luptitude'
_C.TRANSFORM.ZPT = 3630780547.7010174
_C.TRANSFORM.SKYNOISE = [0.01824022, 0.02636513, 0.03169786, 0.06622622, 0.12619147]



def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(config, os.path.join(os.path.dirname(cfg_file), cfg))
        sys.stderr.write('=> merge config form {}\n'.format(cfg_file))
        sys.stderr.flush()
        config.merge_from_file(cfg_file)
        config.freeze()
    
def update_config(config, args):
    if isinstance(args.cfg, str):
        _update_config_from_file(config, args.cfg)
    
    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)
    
    def _check_args(name):
        if hasattr(args, name) and args.name:
            return True
        return False

    if _check_args('model'):
        config.DATA.MODEL = args.model
    if _check_args('data'):
        config.DATA.DATA = args.data
    if _check_args('output'):
        config.DATA.OUTPUT = args.output
    
    config.freeze()

def get_config(args):
    config = _C.clone()
    update_config(config, args)

    return config

