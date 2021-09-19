#!/usr/bin/env python

"""CHRONOS.PY - Automatic isochrone fitting to photometric data

"""

__authors__ = 'David Nidever <dnidever@montana.edu?'
__version__ = '20210919'  # yyyymmdd


import time
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits



def fit(cat,iso):
    """ Automated isochrone fitting to photometric data."""

    # Do a grid search over distance modulues, age, metallicity and extinction

    
