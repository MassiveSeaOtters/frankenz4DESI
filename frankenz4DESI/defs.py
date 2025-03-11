#!/usr/bin/env python

import numpy as np


# ABmagnitude zeropoint in uJy
ABzpt = 10**(-0.4*(-23.9))
# HSC 1-sigma flux depths in uJy
fdepths = np.array([0.01824022, 0.02636513, 0.03169786, 0.06622622, 0.12619147])