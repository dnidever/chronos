#!/usr/bin/env python

"""CHRONOS.PY - Automatic isochrone fitting to photometric data

"""

__authors__ = 'David Nidever <dnidever@montana.edu?'
__version__ = '20210919'  # yyyymmdd

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

def loadiso():
    """ Load all the default isochrone files."""
    ddir = datadir()
    files = glob(ddir+'parsec_*fits.gz')
    nfiles = len(files)
    if nfiles==0:
        raise Exception("No default isochrone files found in "+ddir)
    iso = []
    for f in files:
        iso.append(Table.read(f))
    if len(iso)==1: iso=iso[0]
        
    # Change metallicity and age names for parsec
    iso['AGE'] = 10**iso['LOGAGE'].copy()
    iso['METAL'] = iso['MH']
        
    # Index
    grid = IsoGrid(iso)

    return grid

def loadext():
    """ Load extinctions."""
    ddir = datadir()
    files = glob(ddir+'extinctions.txt')
    nfiles = len(files)
    if nfiles==0:
        raise Exception("No default extinctions file found in "+ddir)
    tab = Table.read(files[0],format='ascii')
    # Turn into dictionary
    ext = {}
    for i in range(len(tab)):
        ext[tab['NAME'][i]] = tab['EXTINCTION'][i]
    return ext
    
def gridparams(ages,metals):
    """ Generator for parameters in grid search."""

    for a in ages:
        for m in metals:
            yield (a,m)

def isocomparison(cphot,isophot,cphoterr=None):
    """ Compare the isochrone to the data."""

    ncat = len(catphot)
    niso = len(isophot)

    # Set up KD-tree
    kdt = cKDTree(isophot)

    # Get distance to closest neighbor for each cphot element
    dist, ind = kdt.query(cphot, k=1, p=2)

    # Goodness of fit metrics
    # Sum of distances
    sumdist = np.sum(dist)

    # Uncertainties
    if cphoterr is not None:
        chisq = np.sum(dist**2/caterr**2)
    else:
        chisq = np.sum(dist**2)        

    return sumdist,chisq


def getiso(grid,pars):
    """ Get single isochrone from age/metallicity grid.  Interpolate if necessary."""

    age,metal = pars

    uages = np.unique(grid['age'])
    umetals = np.unique(grid['metal'])

    # Check that the requested values are inside our grid
    if age<np.min(uages) or age>np.max(uages) or metal<np.min(metals) or metal>np.max(metals):
        raise ValueError('age=%6.2f metal=%6.2f is outside the isochrone grid. %6.2f<age<%6.2f, %6.2f<metal<%6.2f' %
                         (age,meta,np.min(uages),np.max(uages),np.min(umetals),np.max(umetals)))
    
    # Exact match exists
    if age in uages and metal in umetals:
        ind, = np.where((uages==age) & (umetals==metal))
        return grid[ind]
    # Need to interpolate
    else:
        
        print('need to interpolate')
        import pdb; pdb.set_trace()

        
def isoextinct(iso,ext,isonames,extdct):
    """ Apply extinction to the photometry."""

    print('apply extinction')
    import pdb; pdb.set_trace()

    # Just return the photometry array
    
    # photometry data
    phot = []
    for n in isonames:
        mag = iso[n]
        mag += extdict[n]*ext  # add extinction
        phot.append(mag)
    phot = np.vstack(tuple(phot)).T
    return phot
    

def gridsearch(cat,catnames,isonames,grid=None,caterrnames=None,
               ages=None,metals=None,extinctions=None,distmod=None,extdict=None):
    """ Grid search."""
    # extinctions: dictionary of extinction coefficients (Alambda/AV)
    
    # Do a grid search over distance modulues, age, metallicity and extinction
    
    # Default grid values
    if ages is None:
        ages = np.linspace(0.5,12.0,6)
    if metals is None:
        metals = np.linspace(-2.5,0.5,7)
    if extinctions is None:
        extinctions = np.linspace(0.0,1.0,5)
    if distmod is None:
        distmod = np.linspace(0,25.0,11)
    nages = len(ages)
    nmetals = len(metals)
    nextinctions = len(extinctions)
    ndistmod = len(distmod)

    if grid is None:
        grid = loadiso()
    if extdict is None:
        extdict = loadext()
        
    ncat = len(cat)

    # Put observed photometry in 2D array
    cphot = []
    for n in catnames:
        cphot.append(cat[n])
    cphot = np.vstack(tuple(cphot)).T    

    # Uncertainties
    if caterrnames is not None:
        cphoterr = np.zeros(ncat,float)
        for n in caterrnames:
            cphoterr += cat[n]**2
        cphoterr = np.sqrt(cphoterr)
    else:
        cphoterr = np.ones(ncat,float)

    dum=grid(1.05e10,-1.75,names=isonames)
        
    import pdb; pdb.set_trace()
    
    # Grid search
    sumdist = np.zeros((nages,nmetals,nextinctions,ndistmod),float)
    for age,i in enumerate(ages):
        for metal,j in enumerate(metals):
            # Get the isochrone for this value
            isoam = getiso(grid,[age,metal])

            # Extinction and istance modulus search
            for ext,k in enumerate(extinctions):
                # Prep the isochrone for this distance and extinction
                isophot = isoextinct(isoam,ext,isonames,extdict)
                for distm,l in enumerate(distmod):
                    isophot += distm  # add distance modulue

                    # Do the comparison
                    sumdist1,chisq1 = isocomparison(cphot,isophot,cphoterr)
                    sumdist[i,j,k,l] = sumdist1
                    chisq[i,j,k,l] = chisq1


                    # keep track of the smallest distance for each star
                    # if a star never has a good match, then maybe have an option
                    # to trim them out (e.g., horizontal branch, AGB)
                    
    # Get the best match
    
    import pdb; pdb.set_trace()
                    
    
def fit(cat=None,catnames=None,isonames=None,grid=None,caterrnames=None,
        ages=None,metals=None,extinctions=None,distmod=None,extdict=None):
    """ Automated isochrone fitting to photometric data."""

    cat = Table.read('/Users/nidever/papers/chronos/NGC104_gaiaedr3_pmcut.fits')
    catnames = ['BP','RP']
    caterrnames = ['BPERR','RPERR']
    isonames = ['GAIAEDR3_GBPMAG','GAIAEDR3_GRPMAG']
    
    # Do a grid search over distance modulues, age, metallicity and extinction
    best = gridsearch(cat,catnames,isonames,grid=grid,caterrnames=caterrnames,
                      ages=ages,metals=metals,extinctions=extinctions,
                      distmod=distmod,extdict=extdict)
    
    
    # Run MCMC now
