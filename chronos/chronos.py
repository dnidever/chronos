#!/usr/bin/env python

"""CHRONOS.PY - Automatic isochrone fitting to photometric data

"""

__authors__ = 'David Nidever <dnidever@montana.edu?'
__version__ = '20210919'  # yyyymmdd


import time
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from scipy.spatial import cKDTree

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


def getiso(isogrid,pars):
    """ Get single isochrone from age/metallicity grid.  Interpolate if necessary."""

    age,metal = pars

    uages = np.unique(isogrid['age'])
    umetals = np.unique(isogrid['metal'])

    # Check that the requested values are inside our grid
    if age<np.min(uages) or age>np.max(uages) or metal<np.min(metals) or metal>np.max(metals):
        raise ValueError('age=%6.2f metal=%6.2f is outside the isochrone grid. %6.2f<age<%6.2f, %6.2f<metal<%6.2f' %
                         (age,meta,np.min(uages),np.max(uages),np.min(umetals),np.max(umetals)))
    
    # Exact match exists
    if age in uages and metal in umetals:
        ind, = np.where((uages==age) & (umetals==metal))
        return isogrid[ind]
    # Need to interpolate
    else:
        
        print('need to interpolate')
        import pdb; pdb.set_trace()

        
def isoextinct(iso,ext):
    """ Apply extinction to the photometry."""

    print('apply extinction')
    import pdb; pdb.set_trace()

    # Just return the photometry array
    
    # observed data
    cdata = []
    for n in catnames:
        cdata.appen(cat[n])
    cdata = np.vstack(tuple(cdata)).T
    

def grisearch(cat,catnames,isogrid,isonames,caterrnames=None,
              ages=None,metals=None,extinctions=None,distmod=None):
    """ Grid search."""

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

    ncat = len(cat)

    # Put observed photometry in 2D array
    cphot = []
    for n in catnames:
        cphot.appen(cat[n])
    cphot = np.vstack(tuple(cphot)).T    

    # Uncertainties
    if caterrnames is not None:
        cphoterr = np.zeros(ncat,float)
        for n in caterrnames:
            cphoterr += cat[n]**2
        cphoterr = np.sqrt(cphoterr)
    else:
        cphoterr = np.ones(ncat,float)
    
    
    # Grid search
    sumdist = np.zeros((nages,nmetals,nextinctions,ndistmod),float)
    for age,i in enumerate(ages):
        for metal,j in enumerate(metals):
            # Get the isochrone for this value
            isoam = getiso(isogrid,[age,metal])

            # Extinction and istance modulus search
            for ext,k in enumerate(extinctions):
                # Prep the isochrone for this distance and extinction
                isophot = isoextinct(isoam,ext,isonames)      
                for distm,l in enumerate(distmod):
                    isophot += distm  # add distance modulue

                    # Do the comparison
                    sumdist1,chisq1 = isocomparison(cphot,isophot,cphoterr)
                    sumdist[i,j,k,l] = sumdist1
                    chisq[i,j,k,l] = chisq1

    # Get the best match
    
    import pdb; pdb.set_trace()
                    
    
def fit(cat,catnames,isogrid,isonames,caterrnames=None,
        ages=None,metals=None,extinctions=None,distmod=None):
    """ Automated isochrone fitting to photometric data."""
        
    # Do a grid search over distance modulues, age, metallicity and extinction
    best = gridsearch(cat,catnames,isogrid,isonames,caterrnames=caterrnames,
                      ages=ages,metals=metals,extinctions=extinctions,
                      distmod=distmod)
    
    
    # Run MCMC now
