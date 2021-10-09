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
import scipy
from scipy.spatial import cKDTree
import emcee
import corner
import matplotlib
import matplotlib.pyplot as plt
from dlnpyutils import utils as dln
from . import utils,extinction,isochrone,leastsquares as lsq
    
def gridparams(ages,metals):
    """ Generator for parameters in grid search."""

    for a in ages:
        for m in metals:
            yield (a,m)

def autodetectnames(cat,grid,catnames=None,caterrnames=None,isonames=None):
    """ Auto-detect column names to use."""

    gnames = np.char.array(grid.bands).lower()
    cnames = np.char.array(cat.colnames).lower()
    
    # Try to match catalog names with isochrone names
    if catnames is None:
        # Find catalog names that match isochrone column names
        if isonames is None:
            ind1,ind2 = dln.match(cnames,gnames)
            nmatch = len(ind1)
            if nmatch>0:
                catnames = list(np.array(cat.colnames)[ind1])
                isonames = list(np.array(grid.bands)[ind2])

        # isonames input, match catnames to them
        else:
            catnames = []
            for n in isonames:
                if n.lower() in catnames:
                    ind, = np.where(n.lower()==cnames)
                    catnames.append(cat.colnames[ind[0]])
            if len(catnames)==0:
                catnames = None                    

    # No isochrone names, try match them with catnames
    if isonames is None and catnames is not None:
        ind1,ind2 = dln.match(np.char.array(catnames).lower(),gnames)
        nmatch = len(ind1)
        if nmatch!=len(catnames):
            raise ValueError('Not all catnames matched to isochrone band names')
        if nmatch>0:
            isonames = list(np.array(grid.bands)[ind2])
        
    # Check for error columns
    if catnames is not None and caterrnames is None:
        caterrnames = []
        for n in catnames:
            if n.lower()+'_err' in cnames:
                ind, = np.where(n.lower()+'_err'==cnames)
                caterrnames.append(cat.colnames[ind[0]])                
        if len(caterrnames)==0:
            caterrnames = None

    return catnames,caterrnames,isonames
    
            
def isophotprep(iso,names):
    """ Get the isochrone photometry."""
    
    # Put observed photometry in 2D array
    isophot = []
    for n in names:
        isophot.append(iso.data[n])
    isophot = np.vstack(tuple(isophot)).T    
    return isophot
            
def photprep(cat,names,errnames=None,verbose=False):

    ncat = len(cat)
    
    # Put observed photometry in 2D array
    cphot = []
    for n in names:
        cphot.append(cat[n])
    cphot = np.vstack(tuple(cphot)).T    
    
    # Uncertainties
    if errnames is not None:
        cphoterr = np.zeros(ncat,float)
        for n in errnames:
            cphoterr += cat[n]**2
        cphoterr = np.sqrt(cphoterr)
    else:
        cphoterr = np.ones(ncat,float)

    # Only keep good values
    #  check for NaNs or Infs
    cbad = ~np.isfinite(cphot)
    ncbad = np.sum(cbad)
    if ncbad>0:
        cstarbad = np.sum(cbad,axis=1)            
        cphot = cphot[cstarbad==0,:]
        cphoterr = cphoterr[cstarbad==0]
        if verbose:
            print('Trimming out '+str(ncbad)+' stars with bad photometry')

    return cphot,cphoterr


def isocomparison(cphot,isophot,cphoterr=None):
    """ Compare the isochrone to the data."""

    ncat = len(cphot)
    niso = len(isophot)

    # Set up KD-tree
    kdt = cKDTree(isophot)

    # Get distance to closest neighbor for each cphot element
    dist, ind = kdt.query(cphot, k=1, p=2)

    # Goodness of fit metrics
    # Sum of distances
    sumdist = np.sum(dist)
    meddist = np.median(dist)

    # Uncertainties
    if cphoterr is not None:
        chisq = np.sum(dist**2/cphoterr**2)
    else:
        chisq = np.sum(dist**2)        

    # Likelihood
    
        
    return sumdist,meddist,chisq,dist

def printpars(pars,parerr=None):
    npars = len(pars)
    names = ['Age','Metal','Extinction','Distmod']
    units = ['years','dex',' ',' ']
    for i in range(npars):
        if parerr is None:
            err = None
        else:
            err = parerr[i]
        if err is not None:
            print('%-6s =  %8.2f +/- %6.3f %-5s' % (names[i],pars[i],err,units[i]))
        else:
            print('%-6s =  %8.2f %-5s' % (names[i],pars[i],units[i]))


def gridsearch(cat,catnames,grid,isonames,caterrnames=None,
               ages=None,metals=None,extinctions=None,distmod=None,
               fixed=None,extdict=None,verbose=False):
    """
    Grid search.

    Parameters
    ----------
    cat : astropy table
       Observed photometry table/catalog.
    catnames : list
       List of column names for the observed photometry to use.
    grid : IsoGrid object
       Grid of isochrones.
    isonames : list
       List of column names for the isochrone photometry to compare to the observed
        photometry in "catnames".
    caterrnames : list, optional
       List of photometric uncertainty values corresponding to the "catnames" bands.
    ages : list, optional
       List of ages to use in grid search.  Default is np.linspace(0.5,12.0,6)*1e9.
    metals : list, optional
       List of metals to use in grid search. Default is np.linspace(-2.0,0.0,5).
    extinctions : list, optional
       List of extinctions to use in grid search.  Default is np.linspace(0.0,1.0,5).
    distmod : list, optional
       List of distmod to use in grid search.  Default is np.linspace(0,25.0,11).
    fixed : dict, optional
       Dictionary of fixed values to use.
    extdict : dict, optional
       Dictionary of extinction coefficients to use. (A_lambda/A_V).  The column
         names must match the isochrone column names.
    verbose : bool, optional
         Verbose output of the various steps.  This is False by default.       

    Returns
    -------
    bestvals : list
       List of best-fitting parameters [age, metal, ext, distmod].
    bestchisq : float
       Chi-square value for best-fit.

    Example
    -------

    .. code-block:: python

         bestvals, bestchisq = gridsearch(cat,catnames,grid,isonames)

    """
    
    # Default grid values
    if ages is None:
        ages = np.linspace(0.5,12.0,6)*1e9
    if metals is None:
        metals = np.linspace(-2.0,0.0,5)
    if extinctions is None:
        extinctions = np.linspace(0.0,1.0,5)
    if distmod is None:
        distmod = np.linspace(0,25.0,11)

    # Checked any fixed values
    if fixed is not None:
        for n in fixed.keys():
            if n.lower()=='age':
                ages = [fixed[n]]
            elif n.lower()=='logage':
                ages = [10**fixed[n]]
            elif n.lower()=='metal' or n.lower()=='feh' or n.lower()=='fe_h':
                metal = [fixed[n]]
            elif n.lower()=='ext' or n.lower()=='extinction':
                extinctions = [fixed[n]]
            elif n.lower()=='distance' or n.lower()=='dist':
                distmod = [np.log10(fixed[n]*1e3)*5-5]
            elif n.lower()=='distmod':
                distmod = [fixed[n]]
        
    nages = len(ages)
    nmetals = len(metals)
    nextinctions = len(extinctions)
    ndistmod = len(distmod)

    if extdict is None:
        extdict = extinction.load()
        
    ncat = len(cat)

    # Put observed photometry in 2D array
    cphot,cphoterr = photprep(cat,catnames,errnames=caterrnames)
        
    # Grid search
    #------------
    sumdist = np.zeros((nages,nmetals,nextinctions,ndistmod),float)
    meddist = np.zeros((nages,nmetals,nextinctions,ndistmod),float)    
    chisq = np.zeros((nages,nmetals,nextinctions,ndistmod),float)    
    for i,age in enumerate(ages):
        for j,metal in enumerate(metals):
            # Get the isochrone for this value
            iso = grid(age,metal,names=isonames)

            # Extinction and istance modulus search
            for k,ext in enumerate(extinctions):
                iso.ext = ext   # extinction
                for l,distm in enumerate(distmod):
                    iso.distmod = distm  # distance modulus

                    # Get isochrone photometry array
                    isophot = []
                    for n in isonames:
                        isophot.append(iso.data[n])
                    isophot = np.vstack(tuple(isophot)).T  
                    
                    # Do the comparison
                    sumdist1,meddist1,chisq1,dist1 = isocomparison(cphot,isophot,cphoterr)
                    sumdist[i,j,k,l] = sumdist1
                    meddist[i,j,k,l] = meddist1                    
                    chisq[i,j,k,l] = chisq1

                    if verbose:
                        print(i,j,k,l,sumdist1,chisq1)
                    
                    # keep track of the smallest distance for each star
                    # if a star never has a good match, then maybe have an option
                    # to trim them out (e.g., horizontal branch, AGB)
                    
    # Get the best match
    bestind = np.argmin(sumdist)
    bestind2 = np.unravel_index(bestind,sumdist.shape)
    bestvals = [ages[bestind2[0]], metals[bestind2[1]], extinctions[bestind2[2]], distmod[bestind2[3]]]
    bestchisq = chisq.ravel()[bestind]

    return bestvals,bestchisq

def outlier_rejection(cat,catnames,iso,isonames,errnames=None,nsig=3,verbose=False):
    """ Reject outliers using the best-fit isochrone."""

    ncat = len(cat)
    
    # Put observed photometry in 2D array
    cphot,cphoterr = photprep(cat,catnames,errnames=errnames)

    # Get isochrone photometry array
    isophot = []
    for n in isonames:
        isophot.append(iso.data[n])
    isophot = np.vstack(tuple(isophot)).T
                    
    # Do the comparison
    sumdist,meddist,chisq,dist = isocomparison(cphot,isophot,cphoterr)
    sigdist = dln.mad(dist)
    reldist = dist/cphoterr
    medreldist = np.median(reldist)
    sigreldist = dln.mad(reldist)
    #good, = np.where(dist<=nsig*sigdist)
    good, = np.where(reldist<=nsig*sigreldist)    
    nrem = ncat-len(good)
    
    if verbose:
        print('Removed '+str(nrem)+' outlier points from original catalog of '+str(ncat)+' sources')
    
    return cat[good]
    

def allpars(theta,fixpars,fixparvals):
    """ Return values for all 4 parameters dealing with fixed parameters."""
    pars = np.zeros(4,float)
    nfixpars = np.sum(fixpars)
    if nfixpars>0:
        pars[~fixpars] = theta
        pars[fixpars] = fixparvals
    else:
        pars = theta
    return pars

def emcee_lnlike(theta, x, y, yerr, grid, isonames, fixpars, fixparvals):
    """
    This helper function calculates the log likelihood for the MCMC portion of fit().
    
    Parameters
    ----------
    theta : array
      Input parameters [age, metal, ext,distmod].
    x : array
      Array of x-values for y.  Not really used.
    y : array
       Observed photometry array.
    yerr : array
        Uncertainties in the observed photometry data.
    grid : IsoGrid object
        Grid of isochrones.
    isonames : list
        The list of isochrone column names to use.
    fixpars : list
        Boolean list/array indicating if parameters are fixed or not.
    fixparvals: list
        List/array of values to use for fixed parameters.

    Outputs
    -------
    lnlike : float
         The log likelihood value.

    """

    # Get all 4 parameters
    pars = allpars(theta,fixpars,fixparvals)
    
    iso = grid(pars[0],pars[1],pars[2],pars[3],names=isonames)

    # Get isochrone photometry array
    isophot = []
    for n in isonames:
        isophot.append(iso.data[n])
    isophot = np.vstack(tuple(isophot)).T
                    
    # Do the comparison
    sumdist1,meddist1,chisq1,dist1 = isocomparison(y,isophot,yerr)

    return -0.5*chisq1

def emcee_lnprior(theta, grid):
    """
    This helper function calculates the log prior for the MCMC portion of fit().
    It's a flat/uniform prior across the isochrone parameter space covered by the
    isochrone grid.
    
    Parameters
    ----------
    theta : array
       Input parameters [age, metal, ext, distmod].  This needs to be all four.
    grid : IsoGrid object
       Grid of isochrones.

    Outputs
    -------
    lnprior : float
         The log prior value.

    """
    inside = True
    inside &= (theta[0]>=grid.minage and theta[0]<=grid.maxage)
    inside &= (theta[1]>=grid.minmetal and theta[1]<=grid.maxmetal)
    # no distmod limits
    inside &= (theta[3]>=0)
    if inside:
        return 0.0
    return -np.inf
    
def emcee_lnprob(theta, x, y, yerr, grid, isonames, fixpars, fixparvals):
    """
    This helper function calculates the log probability for the MCMC portion of fit().
    
    Parameters
    ----------
    theta : array
      Input parameters [age, metal, ext, distmod].
    x : array
      Array of x-values for y.  Not really used.
    y : array
       Observed photometry.
    yerr : array
        Uncertainties in the observed photometry.
    grid : IsoGrid object
        Grid of isochrones.
    isonames : list
        The list of isochrone column names to use.
    fixpars : list
        Boolean list/array indicating if parameters are fixed or not.
    fixparvals: list
        List/array of values to use for fixed parameters.

    Outputs
    -------
    lnprob : float
         The log probability value, which is the sum of the log prior and the
         log likelihood.

    """
    #print(theta)
    pars = allpars(theta,fixpars,fixparvals)
    lp = emcee_lnprior(pars,grid)
    if not np.isfinite(lp):
        return -np.inf
    return lp + emcee_lnlike(theta, x, y, yerr, grid, isonames, fixpars, fixparvals)

def objectiveiso(theta, y, yerr, grid, isonames, fixpars, fixparvals):

    # Get all 4 parameters
    pars = allpars(theta,fixpars,fixparvals)

    print('objectiveiso: ',pars)
    
    iso = grid(pars[0],pars[1],pars[2],pars[3],names=isonames)

    # Get isochrone photometry array
    isophot = isophotprep(iso,isonames)
                    
    # Do the comparison
    sumdist1,meddist1,chisq1,dist1 = isocomparison(y,isophot,yerr)

    return 0.5*chisq1


def funiso(theta,cphot,cphoterr,grid,isonames,fixpars,fixparvals,verbose=False):
    """ Return the function and gradient."""

    pars = allpars(theta,fixpars,fixparvals)
    
    ncat = len(cphoterr)
    nfreepars = np.sum(~fixpars)
    grad = np.zeros(nfreepars,float)    
    pcount = 0

    if verbose:
        print('funiso: ',pars)
    
    # Original model
    iso0 = grid(*pars)
    isophot0 = isophotprep(iso0,isonames)
    sumdist0,meddist0,chisq0,dist0 = isocomparison(cphot,isophot0,cphoterr)
    lnlike0 = 0.5*chisq0

    # Bad input parameter values
    if (pars[0]<grid.minage or pars[0]>grid.maxage) or (pars[1]<grid.minmetal or pars[1]>grid.maxmetal) or \
       (pars[2]<0):
        return np.inf, np.array(4,float)
    
    
    # Derivative in age
    if fixpars[0]==False:
        pars1 = np.array(pars).copy()
        step = 0.05*pars[0]
        if pars1[0]+step>grid.maxage:
            step -= step     
        pars1[0] += step
        iso1 = grid(*pars1)
        isophot1 = isophotprep(iso1,isonames)
        sumdist1,meddist1,chisq1,dist1 = isocomparison(cphot,isophot1,cphoterr)
        lnlike1 = 0.5*chisq1        
        grad[pcount] = (lnlike1-lnlike0)/step
        pcount += 1
        
    # Derivative in metallicity
    if fixpars[1]==False:
        pars2 = np.array(pars).copy()
        step = 0.05
        if pars2[1]+step>grid.maxmetal:
            step -= step     
        pars2[1] += step
        iso2 = grid(*pars2)
        isophot2 = isophotprep(iso2,isonames)
        sumdist2,meddist2,chisq2,dist2 = isocomparison(cphot,isophot2,cphoterr)
        lnlike2 = 0.5*chisq2
        grad[pcount] = (lnlike2-lnlike0)/step        
        pcount += 1
        
    # Derivative in extinction
    if fixpars[2]==False:
        iso3 = iso0.copy()
        step = 0.05
        iso3.ext += step
        isophot3 = isophotprep(iso3,isonames)
        sumdist3,meddist3,chisq3,dist3 = isocomparison(cphot,isophot3,cphoterr)
        lnlike3 = 0.5*chisq3
        #jac[:,pcount] = dist3-dist0
        #jac[:,pcount] = (sumdist3-sumdist0)/step
        grad[pcount] = (lnlike3-lnlike0)/step        
        pcount += 1
    
    # Derivative in distmod
    if fixpars[3]==False:
        iso4 = iso0.copy()
        step = 0.05
        iso4.distmod += step
        isophot4 = isophotprep(iso4,isonames)
        sumdist4,meddist4,chisq4,dist4 = isocomparison(cphot,isophot4,cphoterr)
        lnlike4 = 0.5*chisq4        
        #jac[:,pcount] = dist4-dist0
        #jac[:,pcount] = (sumdist4-sumdist0)/step
        grad[pcount] = (lnlike4-lnlike0)/step        
        pcount += 1
        
    return lnlike0,grad


def gradiso(theta,cphot,cphoterr,grid,isonames,fixpars,fixparvals):
    """ Calculate gradient for Isochrone fits."""

    pars = allpars(theta,fixpars,fixparvals)
    
    ncat = len(cphoterr)
    nfreepars = np.sum(~fixpars)
    grad = np.zeros(nfreepars,float)    
    pcount = 0

    print('gradiso: ',pars)
    
    # Original model
    iso0 = grid(*pars)
    isophot0 = isophotprep(iso0,isonames)
    sumdist0,meddist0,chisq0,dist0 = isocomparison(cphot,isophot0,cphoterr)
    lnlike0 = -0.5*chisq0
    
    # Derivative in age
    if fixpars[0]==False:
        pars1 = np.array(pars).copy()
        step = 0.05*pars[0]
        if pars1[0]+step>grid.maxage:
            step -= step        
        pars1[0] += step
        iso1 = grid(*pars1)
        isophot1 = isophotprep(iso1,isonames)
        sumdist1,meddist1,chisq1,dist1 = isocomparison(cphot,isophot1,cphoterr)
        lnlike1 = -0.5*chisq1        
        #jac[:,pcount] = dist1-dist0
        #jac[:,pcount] = (sumdist1-sumdist0)/step
        grad[pcount] = (lnlike1-lnlike0)/step
        pcount += 1
        
    # Derivative in metallicity
    if fixpars[1]==False:
        pars2 = np.array(pars).copy()
        step = 0.05
        if pars2[1]+step>grid.maxmetal:
            step -= step
        pars2[1] += step
        iso2 = grid(*pars2)
        isophot2 = isophotprep(iso2,isonames)
        sumdist2,meddist2,chisq2,dist2 = isocomparison(cphot,isophot2,cphoterr)
        lnlike2 = -0.5*chisq2
        #jac[:,pcount] = dist2-dist0
        #jac[:,pcount] = (sumdist2-sumdist0)/step
        grad[pcount] = (lnlike2-lnlike0)/step        
        pcount += 1
        
    # Derivative in extinction
    if fixpars[2]==False:
        iso3 = iso0.copy()
        step = 0.05
        iso3.ext += step
        isophot3 = isophotprep(iso3,isonames)
        sumdist3,meddist3,chisq3,dist3 = isocomparison(cphot,isophot3,cphoterr)
        lnlike3 = -0.5*chisq3
        #jac[:,pcount] = dist3-dist0
        #jac[:,pcount] = (sumdist3-sumdist0)/step
        grad[pcount] = (lnlike3-lnlike0)/step        
        pcount += 1
    
    # Derivative in distmod
    if fixpars[3]==False:
        iso4 = iso0.copy()
        step = 0.05
        iso4.distmod += step
        isophot4 = isophotprep(iso1,isonames)
        sumdist4,meddist4,chisq4,dist4 = isocomparison(cphot,isophot4,cphoterr)
        lnlike4 = -0.5*chisq4        
        #jac[:,pcount] = dist4-dist0
        #jac[:,pcount] = (sumdist4-sumdist0)/step
        grad[pcount] = (lnlike4-lnlike0)/step        
        pcount += 1
        
    return -grad

def hessiso(theta,cphot,cphoterr,grid,isonames,fixpars,fixparvals,diag=False):
    """ Calculate hessian matrix, second derivaties wrt parameters."""

    pars = allpars(theta,fixpars,fixparvals)
    
    ncat = len(cphoterr)
    nfreepars = np.sum(~fixpars)
    freeparsind, = np.where(fixpars==False)
    hess = np.zeros((nfreepars,nfreepars),float)    


    # Original model
    iso0 = grid(*pars)
    isophot0 = isophotprep(iso0,isonames)
    sumdist0,meddist0,chisq0,dist0 = isocomparison(cphot,isophot0,cphoterr)
    lnlike0 = 0.5*chisq0

    steps = [0.05*pars[0],0.05,0.05,0.05]
    
    # Loop over all free parameters
    for i in range(nfreepars):
        ipar = freeparsind[i]
        istep = steps[ipar]
        # Make sure steps don't go beyond boundaries
        if ipar==0 and (pars[0]+2*istep)>grid.maxage:
            istep = -istep
        if ipar==1 and (pars[1]+2*istep)>grid.maxmetal:
            istep = -istep
        # Second loop
        for j in np.arange(0,i+1):
            jpar = freeparsind[j]
            jstep = steps[ipar]
            # Make sure steps don't go beyond boundaries
            if jpar==0 and (pars[0]+2*jstep)>grid.maxage:
                jstep = -jstep
            if jpar==1 and (pars[1]+2*jstep)>grid.maxmetal:
                jstep = -jstep
            
            # Calculate the second derivative wrt i and j

            # Second derivative of same parameter
            #   Derivative one step forward and two steps forward
            #   then take the derivative of these two derivatives            
            if i==j:
                # First first-derivative
                pars1 = pars.copy()
                pars1[ipar] += istep
                if ipar<2:
                    iso1 = grid(*pars1)
                elif ipar==2:
                    iso1 = iso0.copy()
                    iso1.ext += istep
                elif ipar==3:
                    iso1 = iso0.copy()
                    iso1.distmod += istep                    
                isophot1 = isophotprep(iso1,isonames)                    
                sumdist1,meddist1,chisq1,dist1 = isocomparison(cphot,isophot1,cphoterr)                    
                lnlike1 = 0.5*chisq1
                deriv1 = (lnlike1-lnlike0)/istep
                # Second first-derivative
                pars2 = pars.copy()
                pars2[ipar] += 2*istep
                if ipar<2:
                    iso2 = grid(*pars2)
                elif ipar==2:
                    iso2 = iso0.copy()
                    iso2.ext += 2*istep
                elif ipar==3:
                    iso2 = iso0.copy()
                    iso2.distmod += 2*istep                    
                isophot2 = isophotprep(iso2,isonames)                    
                sumdist2,meddist2,chisq2,dist2 = isocomparison(cphot,isophot2,cphoterr)                    
                lnlike2 = 0.5*chisq2
                deriv2 = (lnlike2-lnlike1)/istep
                # Second derivative
                deriv2nd = (deriv2-deriv1)/istep
                hess[i,j] = deriv2nd

            # Two different parameters
            #   Derivative in i at current position
            #   Derivative in i at current position plus one step in j
            #   take derivate of these two derivatives
            else:
                # Only want diagonal elements
                if diag:
                    continue
                
                # First first-derivative
                # derivative in i at current j position
                pars1 = pars.copy()
                pars1[ipar] += istep
                if ipar<2:
                    iso1 = grid(*pars1)
                elif ipar==2:
                    iso1 = iso0.copy()
                    iso1.ext += istep
                elif ipar==3:
                    iso1 = iso0.copy()
                    iso1.distmod += istep                    
                isophot1 = isophotprep(iso1,isonames)                    
                sumdist1,meddist1,chisq1,dist1 = isocomparison(cphot,isophot1,cphoterr)                    
                lnlike1 = 0.5*chisq1
                deriv1 = (lnlike1-lnlike0)/istep
                # Second first-derivative
                # derivatve in i at current position plus one step in j
                
                # Likelihood at current position plust one step in j
                pars2 = pars.copy()
                pars2[jpar] += jstep
                if jpar<2:
                    iso2 = grid(*pars2)
                elif jpar==2:
                    iso2 = iso0.copy()
                    iso2.ext += jstep
                elif jpar==3:
                    iso2 = iso0.copy()
                    iso2.distmod += jstep                       
                isophot2 = isophotprep(iso2,isonames)                    
                sumdist2,meddist2,chisq2,dist2 = isocomparison(cphot,isophot2,cphoterr)                    
                lnlike2 = 0.5*chisq2
                # Likelihood at current position plus one step in i and j
                pars3 = pars.copy()
                pars3[ipar] += istep
                pars3[jpar] += jstep
                if ipar>=2 and jpar>=2:
                    if ipar==2:
                        iso3.ext += istep
                    elif ipar==3:
                        iso3.distmod += istep
                    if jpar==2:
                        iso3.ext += jstep
                    elif jpar==3:
                        iso3.distmod += jstep
                else:
                    iso3 = grid(*pars3)                        
                isophot3 = isophotprep(iso3,isonames)                    
                sumdist3,meddist3,chisq3,dist3 = isocomparison(cphot,isophot3,cphoterr)                    
                lnlike3 = 0.5*chisq3                
                deriv2 = (lnlike3-lnlike2)/istep
                # Second derivative
                deriv2nd = (deriv2-deriv1)/jstep
                hess[i,j] = deriv2nd
                hess[j,i] = deriv2nd                

    return hess
        
        
def fit_mle(cat,catnames,grid,isonames,initpar,caterrnames=None,fixed=None,verbose=False):
    """ Isochrone fitting using maximum likelihood estimation (MLE)."""
    
    ncat = len(cat)

    cphot,cphoterr = photprep(cat,catnames,caterrnames)
    
    # Checked any fixed values
    fixpars = np.zeros(4,bool)
    if fixed is not None:
        for n in fixed.keys():
            if n.lower()=='age':
                initpar[0] = fixed[n]
                fixpars[0] = True
            elif n.lower()=='logage':
                initpar[0] = 10**fixed[n]
                fixpars[0] = True                
            elif n.lower()=='metal' or n.lower()=='feh' or n.lower()=='fe_h':
                initpar[1] = fixed[n]
                fixpars[1] = True                
            elif n.lower()=='ext' or n.lower()=='extinction':
                initpar[2] = fixed[n]
                fixpars[2] = True                
            elif n.lower()=='distance' or n.lower()=='dist':
                initpar[3] = np.log10(fixed[n]*1e3)*5-5
                fixpars[3] = True                
            elif n.lower()=='distmod':
                initpar[3] = fixed[n]
                fixpars[4] = True
    nfixpars = np.sum(fixpars)
    nfreepars = np.sum(~fixpars)
    freeparsind, = np.where(fixpars==False)

    if nfixpars>0:
        fixparsind, = np.where(fixpars==True)        
        fixparvals = np.zeros(nfixpars,float)
        fixparvals[:] = np.array(initpar)[fixparsind]
        initpar = np.delete(initpar,fixparsind)
    else:
        fixparvals = []

    # Bounds
    lbounds = np.zeros(4,float)
    lbounds[0] = grid.minage
    lbounds[1] = grid.minmetal
    lbounds[2] = 0.0
    lbounds[3] = -np.inf # None
    ubounds = np.zeros(4,float)
    ubounds[0] = grid.maxage
    ubounds[1] = grid.maxmetal
    ubounds[2] = np.inf  # None
    ubounds[3] = np.inf  # None
    if nfixpars>0:
        lbounds = np.delete(lbounds,fixparsind)
        ubounds = np.delete(ubounds,fixparsind)        
    bounds = list(zip(lbounds,ubounds))
        
    
    # Use scipy.optimize.minimize
    res = scipy.optimize.minimize(funiso,initpar,jac=True,method='L-BFGS-B',options={'ftol':2e-3,'gtol': 2e-3,'maxiter':50},
                                  args=(cphot,cphoterr,grid,isonames,fixpars,fixparvals),bounds=bounds)    
    theta = res.x
    pars = allpars(theta,fixpars,fixparvals)

    # Best model
    iso = grid(*pars)
    isophot = isophotprep(iso,isonames)
    sumdist,meddist,chisq,dist = isocomparison(cphot,isophot,cphoterr)

    # Variance of an ML estimator is the inverse of the Fisher information matrix
    # http://www.sherrytowers.com/mle_introduction.pdf, pg. 7+8
    # https://stats.stackexchange.com/questions/68080/basic-question-about-fisher-information-matrix-and-relationship-to-hessian-and-s
    # var = [I]^-1    (means 1/[I])
    # information matrix is negative of expectation value of Hessian matrix
    # [I] = -E[H]
    # Hessian matrix is the matrix of second derivatives of the likelihood
    # with respect to the parameters
    # Therefore,
    # var = (-E[H])^-1
    # Standard errors of the estimators are the sqrt() of the diagnoal terms
    # in the variance-covariance matrix.
    hess = hessiso(theta,cphot,cphoterr,grid,isonames,fixpars,fixparvals,diag=True)
    thetaerr = np.sqrt(1/np.diag(hess))
    parerr = np.zeros(4,float)
    parerr[freeparsind] = thetaerr

    if verbose:
        printpars(pars,parerr) 
    
    return pars,parerr,chisq


def fit_mcmc(cat,catnames,grid,isonames,caterrnames=None,initpar=None,
             fixed=None,steps=100,extdict=None,cornername=None,verbose=False):
    """
    Fit isochrone to the observed photometry using MCMC.
    
    Parameters
    ----------
    cat : astropy table
       Observed photometry table/catalog.
    catnames : list
       List of column names for the observed photometry to use.
    grid : IsoGrid object
       Grid of isochrones.
    isonames : list
       List of column names for the isochrone photometry to compare to the observed
        photometry in "catnames".
    caterrnames : list, optional
       List of photometric uncertainty values corresponding to the "catnames" bands.
    initpar : numpy array, optional
         Initial estimate for [age, metal, ext, distmod], optional.
    fixed : dict, optional
       Dictionary of fixed values to use.
    steps : int, optional
         Number of steps to use.  Default is 100.
    extdict : dict, optional
         Dictionary of extinction coefficients to use (A_lambda/A_V).
    cornername : string, optional
         Output filename for the corner plot.  If a corner plot is requested, then the
         minimum number of steps used is 500.
    verbose : bool, optional
         Verbose output of the various steps.  This is False by default.

    Returns
    -------
    out : table
       Table of best-fitting values and uncertainties.
    mciso : float
       Best-fitting isochrone.

    Example
    -------

    .. code-block:: python

         out, mciso = fit_mcmc(cat,catnames,grid,isonames)


    """
    
    # Put observed photometry in 2D array
    ncat = len(cat)
    cphot,cphoterr = photprep(cat,catnames,errnames=caterrnames)

    # Initial guesses
    if initpar is None:
        initpar = [5e9, -0.5, 0.1, 15.0]
    
    # Checked any fixed values
    fixpars = np.zeros(4,bool)
    if fixed is not None:
        for n in fixed.keys():
            if n.lower()=='age':
                initpar[0] = fixed[n]
                fixpars[0] = True
            elif n.lower()=='logage':
                initpar[0] = 10**fixed[n]
                fixpars[0] = True                
            elif n.lower()=='metal' or n.lower()=='feh' or n.lower()=='fe_h':
                initpar[1] = fixed[n]
                fixpars[1] = True                
            elif n.lower()=='ext' or n.lower()=='extinction':
                initpar[2] = fixed[n]
                fixpars[2] = True                
            elif n.lower()=='distance' or n.lower()=='dist':
                initpar[3] = np.log10(fixed[n]*1e3)*5-5
                fixpars[3] = True                
            elif n.lower()=='distmod':
                initpar[3] = fixed[n]
                fixpars[4] = True
    nfixpars = np.sum(fixpars)
    nfreepars = np.sum(~fixpars)
    
    # Set up the MCMC sampler
    ndim, nwalkers = nfreepars, 20
    delta = np.array([initpar[0]*0.25, 0.2, 0.2, 0.2])
    if nfixpars>0:
        fixparsind, = np.where(fixpars==True)        
        fixparvals = np.zeros(nfixpars,float)
        fixparvals[:] = np.array(initpar)[fixparsind]
        delta = np.delete(delta,fixparsind)
        initpar = np.delete(initpar,fixparsind)
    else:
        fixparvals = []
    pos = [initpar + delta*np.random.randn(ndim) for i in range(nwalkers)]
    
    x = np.arange(ncat)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, emcee_lnprob,
                                    args=(x,cphot,cphoterr,grid,isonames,fixpars,fixparvals))

    if cornername is not None: steps=np.maximum(steps,500)  # at least 500 steps
    out = sampler.run_mcmc(pos, steps)
    
    samples = sampler.chain[:, np.int(steps/2):, :].reshape((-1, ndim))

    # Get the median and stddev values
    pars = np.zeros(ndim,float)
    parerr = np.zeros(ndim,float)
    if verbose is True: print('MCMC values:')
    names = ['age','metal','extinction','distmod']
    for i in range(ndim):
        t = np.percentile(samples[:,i],[16,50,84])
        pars[i] = t[1]
        parerr[i] = (t[2]-t[0])*0.5
    if verbose is True:
        allp = allpars(pars,fixpars,fixparvals)
        printpars(allp,parerr)
        
    # The maximum likelihood parameters
    bestind = np.unravel_index(np.argmax(sampler.lnprobability),sampler.lnprobability.shape)
    pars_ml = sampler.chain[bestind[0],bestind[1],:]

    bestpars = allpars(pars_ml,fixpars,fixparvals)
    mciso = grid(bestpars[0],bestpars[1],bestpars[2],bestpars[3])
    isophot = []
    for n in isonames:
        isophot.append(mciso.data[n])
    isophot = np.vstack(tuple(isophot)).T
    mcsumdist,mcmeddist,mcchisq,mcdist = isocomparison(cphot,isophot,cphoterr)
    
    # Put it into the output structure
    dtype = np.dtype([('pars',float,4),('pars_ml',float,4),('parerr',float,4),
                      ('fixed',int,4),('maxlnlike',float),('chisq',float)])
    out = np.zeros(1,dtype=dtype)
    out['fixed'] = 0
    if nfixpars>0:
        freeparsind, = np.where(fixpars==False)
        fixparsind, = np.where(fixpars==True)                
        out['fixed'][0][fixparsind] = 1
        out['pars'][0][fixparsind] = fixparvals
        out['pars_ml'][0][fixparsind] = fixparvals        
        out['pars'][0][freeparsind] = pars        
        out['pars_ml'][0][freeparsind] = pars_ml
        out['parerr'][0][freeparsind] = parerr                
    else:
        out['pars'] = pars
        out['pars_ml'] = pars_ml    
        out['parerr'] = parerr
    out['maxlnlike'] = sampler.lnprobability[bestind[0],bestind[1]]
    out['chisq'] = mcchisq
    
    # Corner plot
    if cornername is not None:
        matplotlib.use('Agg')
        fig = corner.corner(samples, labels=["Age", "Metal", "Extinction", "Distmod"], truths=pars)
        plt.savefig(cornername)
        plt.close(fig)
        if verbose: print('Corner plot saved to '+cornername)
        
    return out,mciso

def cmdfigure(figfile,cat,catnames,iso,isonames,out,annotlabels=None,figsize=10,title=None,verbose=False):

    """
    Make diagnostic figure.

    Parameters
    ----------
    figfile : str
       Output figure filename.
    cat : astropy table
       Observed photometry table/catalog.
    catnames : list
       List of column names for the observed photometry to use.
    grid : IsoGrid object
       Grid of isochrones.
    isonames : list
       List of column names for the isochrone photometry to compare to the observed
        photometry in "catnames".
    out : table
       Catalog of best-fitting values to use for the annotations.
    annotlabels : list, optional
       The list of labels to use for the annotation.  Default is ['age','metal','ext','distmod'].
    figsize : float, optional
       Figure size to use.  Default is 10 inches.
    title : str, optional
       The figure plot title.  Default is "Chronos Isochrone Fit".
    verbose : boolean, optional
       Verbose output.  Default is True.

    Returns
    -------
    Figure is saved to figfile.

    Example
    -------
    .. code-block:: python
           
         cmdfigure(figfile,cat,catnames,grid,isonames,out,verbose=True,annotlabels=annotlabels)

    """
    
    if annotlabels is None:
        annotlabels = ['age','metal','ext','distmod']
    if title is None:
        title = 'Chronos Isochrone Fit'
    matplotlib.use('Agg')
    if os.path.exists(figfile): os.remove(figfile)
    nlegcol = 2

    fig,ax = plt.subplots()
    fig.set_figheight(figsize)
    fig.set_figwidth(figsize*0.5)

    # Make a color of the first two bands
    # color = band1-band2
    # magnitude = band2
    cphot,cerrphot = photprep(cat,catnames)
    catcolor = cphot[:,0]-cphot[:,1]
    catmag = cphot[:,1]
    isocolor = iso.data[isonames[0]].data-iso.data[isonames[1]].data
    isomag = iso.data[isonames[1]].data
    norm = matplotlib.colors.LogNorm()
    plt.hist2d(catcolor,catmag,label='Data',bins=100,cmap='gray_r',norm=norm)

    plt.colorbar(label='Nstars',orientation='horizontal',anchor=(0.5,1.0),pad=0.08)
    #plt.scatter(catcolor,catmag,c='b',label='Data',s=5)
    # plotting isochrone, deal with gaps
    plt.plot(isocolor,isomag,'r',label='Isochrone',linewidth=1,alpha=0.8)
    leg = ax.legend(loc='upper left', frameon=True, framealpha=0.8, ncol=nlegcol)
    plt.xlabel('Color ('+catnames[0]+'-'+catnames[1]+')')
    plt.ylabel('Magnitude ('+catnames[1]+')')
    xr = dln.minmax(catcolor)
    yr = dln.minmax(catmag)
    yr = [yr[0]-dln.valrange(yr)*0.05,yr[1]+dln.valrange(yr)*0.05]
    yr = np.flip(yr)
    plt.xlim(xr)
    plt.ylim(yr)
    plt.title(title)
    string = r'Age=%5.2e$\pm$%5.1e  metal=%5.2f$\pm$%5.2f' % \
               (out['age'][0],out['ageerr'][0],out['metal'][0],out['metalerr'][0])
    string += '\n'
    string += 'ext=%5.2f$\pm$%5.2f  distmod=%5.2f$\pm$%5.2f' % \
               (out['ext'][0],out['exterr'][0],out['distmod'][0],out['distmoderr'][0])
    ax.annotate(string,xy=(np.mean(xr),yr[0]-dln.valrange(yr)*0.05),ha='center',color='black')
    plt.savefig(figfile,bbox_inches='tight')
    plt.close(fig)
    if verbose is True: print('Figure saved to '+figfile)
    
    
def fit(cat,catnames=None,isonames=None,grid=None,caterrnames=None,
        ages=None,metals=None,extinctions=None,distmod=None,initpar=None,
        fixed=None,extdict=None,msteps=100,cornername=None,figfile=None,
        mcmc=False,reject=False,nsigrej=3.0,verbose=False):
    """
    Automated isochrone fitting to photometric data.

    Parameters
    ----------
    cat : astropy table
       Observed photometry table/catalog.
    catnames : list
       List of column names for the observed photometry to use.
    isonames : list
       List of column names for the isochrone photometry to compare to the observed
        photometry in "catnames".
    grid : IsoGrid object, optional
       Grid of isochrones.
    caterrnames : list, optional
       List of photometric uncertainty values corresponding to the "catnames" bands.
    ages : list, optional
       List of ages to use in grid search.  Default is np.linspace(0.5,12.0,6)*1e9.
    metals : list, optional
       List of metals to use in grid search. Default is np.linspace(-2.0,0.0,5).
    extinctions : list, optional
       List of extinctions to use in grid search.  Default is np.linspace(0.0,1.0,5).
    distmod : list, optional
       List of distmod to use in grid search.  Default is np.linspace(0,25.0,11).
    initpar : list
       List of initial estimates for [age, metal, ext, distmod].
    fixed : dict, optional
       Dictionary of fixed values to use.
    extdict : dict, optional
       Dictionary of extinction coefficients to use. (A_lambda/A_V).  The column
         names must match the isochrone column names.
    mcmc : bool, optional
       Run MCMC for better uncertainty estimation.  Default is False.
    reject : bool, optional
       Reject outliers.  Default is False.
    nsigrej : float, optional
       Outlier rejection Nsigma.  Default is 3.0.
    msteps : int, optional
       Number of MCMC steps.  Default is 100.
    cornername : string, optional
         Filename for the corner plot.
    figfile : str, optional
       Filename for the dianostic CMD plot.
    verbose : bool, optional
       Verbose output to the screen.  Default is False.
    
    Returns
    -------
    out : table
       Catalog of best-fitting values and uncertainties
    bestiso : Isochrone object
       Best-fitting isochrone.

    Example
    -------

    .. code-block:: python

         out,bestiso = fit(cat,catnames,grid,isonames)

    """

    if grid is None:
        grid = isochrone.load()
    if extdict is None:
        extdict = extinction.load()

    # don't necessarily have to input the catnames and isonames if you give the catalog column names the
    # same names as the isochrone names.  Can then match them automatically.

    # Automatically detect photometric bands
    if isonames is None or catnames is None:
        catnames,caterrnames,isonames = autodetectnames(cat,grid,catnames=catnames,
                                                        caterrnames=caterrnames,isonames=isonames)

    # Check names
    if catnames is None:
        raise ValueError('Need catnames')    
    if isonames is None:
        raise ValueError('Need isonames')
    if len(catnames)!=len(isonames):
        raise ValueError('Length of catnames and isonames are not equal')
        
    if verbose:
        print('Fitting isochrones to catalog of '+str(len(cat))+' sources')
        print('Photometry columns: '+', '.join(catnames))
        if caterrnames is not None:
            print('Photometric uncertainty columns: '+', '.join(caterrnames))
        print('Isochrone columns: '+', '.join(isonames))

    if initpar is not None and verbose:
        print(' ')
        print('Using initial parameter estimates:')
        printpars(initpar)
        
    # NEED TO HAVE FINER SAMPLING OF ISOCHRONE!!!
    # If the age+metallicity are fixed, then you don't need to interpolate at all!  Just use single isochrone.
    # If age or metallicity are fixed, the you could interpolate the grid in that one dimension and
    #  the rest would be much faster.
        
    # Do a grid search over distance modulues, age, metallicity and extinction
    if initpar is None:
        if verbose:
            print(' ')
            print('Performing grid search')
        bestval,chisq = gridsearch(cat,catnames,grid,isonames,caterrnames=caterrnames,
                                   ages=ages,metals=metals,extinctions=extinctions,
                                   distmod=distmod,fixed=fixed,extdict=extdict)
    else:
        bestval = initpar

    # Maximum likelihood estimation
    if verbose:
        print(' ')
        print('Performing maximum likelihood estimation')
    lsqpars,lsqparerror,lsqchisq = fit_mle(cat,catnames,grid,isonames,bestval,caterrnames=caterrnames,
                                           fixed=fixed,verbose=verbose)

    # Outlier rejection
    if reject:
        bestiso = grid(bestval[0],bestval[1],bestval[2],bestval[3])
        newcat = outlier_rejection(cat,catnames,bestiso,isonames,
                                   errnames=caterrnames,nsig=nsigrej,
                                   verbose=verbose)
        orig = cat.copy()
        cat = newcat

        lsqpars1 = lsqpars.copy()
        lsqparerror1 = lsqparerror.copy()
        lsqchisq1 = lsqchisq

        if verbose:
            print(' ')
            print('Performing maximum likelihood estimation again')
        lsqpars,lsqparerror,lsqchisq = fit_mle(cat,catnames,grid,isonames,lsqpars,caterrnames=caterrnames,
                                               fixed=fixed,verbose=verbose)


    # Run MCMC now
    if mcmc:
        if verbose:
            print(' ')
            print('Running MCMC')
        mcout,mciso = fit_mcmc(cat,catnames,grid,isonames,caterrnames=caterrnames,
                               initpar=bestval,fixed=fixed,extdict=extdict,steps=msteps,
                               cornername=cornername,verbose=verbose)
        fpars = mcout['pars_ml'][0]
        fparerror = mcout['parerr'][0]
        fchisq = mcout['chisq']
    else:
        fpars = lsqpars
        fparerror = lsqparerror
        fchisq = lsqchisq
    bestiso = grid(*fpars)
    
    if verbose is True:
        print(' ')
        print('Final parameters:')
        printpars(fpars,fparerror)
        print('chisq = %5.2f' % fchisq)
    dtype = np.dtype([('age',np.float32),('ageerr',np.float32),('metal',np.float32),('metalerr',np.float32),
                      ('ext',np.float32),('exterr',np.float32),('distmod',np.float32),('distmoderr',np.float32),
                      ('distance',np.float32),('chisq',np.float32)])
    out = np.zeros(1,dtype=dtype)
    out['age'] = fpars[0]
    out['ageerr'] = fparerror[0]  
    out['metal'] = fpars[1]
    out['metalerr'] = fparerror[1]  
    out['ext'] = fpars[2]
    out['exterr'] = fparerror[2]
    out['distmod'] = fpars[3]
    out['distmoderr'] = fparerror[3]
    out['distance'] = 10**((out['distmod']+5)/5.)/1e3
    out['chisq'] = fchisq
    
    # Make output plot
    if figfile is not None:
        cmdfigure(figfile,cat,catnames,bestiso,isonames,out,verbose=verbose)

    return out,bestiso
