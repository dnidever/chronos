#!/usr/bin/env python

"""UTILS.PY - Utility functions

"""

__authors__ = 'David Nidever <dnidever@montana.edu?'
__version__ = '20210928'  # yyyymmdd

import os
import time
import numpy as np
from glob import glob

def datadir():
    """ Return the data directory name."""
    fil = os.path.abspath(__file__)
    codedir = os.path.dirname(fil)
    datadir = codedir+'/data/'
    return datadir

# Split a filename into directory, base and fits extensions
def splitfilename(filename):
    """ Split filename into directory, base and extensions."""
    fdir = os.path.dirname(filename)
    base = os.path.basename(filename)
    exten = ['.fit','.fits','.fit.gz','.fits.gz','.fit.fz','.fits.fz']
    for e in exten:
        if base[-len(e):]==e:
            base = base[0:-len(e)]
            ext = e
            break
    return (fdir,base,ext)
