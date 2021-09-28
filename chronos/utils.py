#!/usr/bin/env python

"""UTILS.PY - Utility functions

"""

__authors__ = 'David Nidever <dnidever@montana.edu?'
__version__ = '20210928'  # yyyymmdd

import os
import time
import numpy as np
from glob import glob
from astropy.wcs import WCS
from astropy.io import fits
from astropy.table import Table
from scipy.spatial import cKDTree
from .isogrid import IsoGrid

def datadir():
    """ Return the data directory name."""
    fil = os.path.abspath(__file__)
    codedir = os.path.dirname(fil)
    datadir = codedir+'/data/'
    return datadir
