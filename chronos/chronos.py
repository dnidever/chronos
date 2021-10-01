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
import emcee
import corner
import matplotlib
import matplotlib.pyplot as plt
from dlnpyutils import utils as dln
from . import utils,extinction,isochrone
    
def gridparams(ages,metals):
    """ Generator for parameters in grid search."""

    for a in ages:
        for m in metals:
            yield (a,m)

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
        
    return sumdist,meddist,chisq

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
               fixed=None,extdict=None):
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
                    sumdist1,meddist1,chisq1 = isocomparison(cphot,isophot,cphoterr)
                    sumdist[i,j,k,l] = sumdist1
                    meddist[i,j,k,l] = meddist1                    
                    chisq[i,j,k,l] = chisq1

                    print(i,j,k,l,sumdist1,chisq1)
                    
                    #import pdb; pdb.set_trace()
                    
                    # keep track of the smallest distance for each star
                    # if a star never has a good match, then maybe have an option
                    # to trim them out (e.g., horizontal branch, AGB)
                    
    # Get the best match
    bestind = np.argmin(sumdist)
    bestind2 = np.unravel_index(bestind,sumdist.shape)
    bestvals = [ages[bestind2[0]], metals[bestind2[1]], extinctions[bestind2[2]], distmod[bestind2[3]]]
    bestchisq = chisq.ravel()[bestind]

    return bestvals,bestchisq


def emcee_lnlike(theta, x, y, yerr, grid, isonames):
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

    Outputs
    -------
    lnlike : float
         The log likelihood value.

    """
    iso = grid(theta[0],theta[1],theta[2],theta[3],names=isonames)

    # Get isochrone photometry array
    isophot = []
    for n in isonames:
        isophot.append(iso.data[n])
    isophot = np.vstack(tuple(isophot)).T
                    
    # Do the comparison
    sumdist1,meddist1,chisq1 = isocomparison(y,isophot,yerr)

    return -0.5*chisq1

def emcee_lnprior(theta, grid):
    """
    This helper function calculates the log prior for the MCMC portion of fit().
    It's a flat/uniform prior across the isochrone parameter space covered by the
    isochrone grid.
    
    Parameters
    ----------
    theta : array
       Input parameters [teff, logg, feh, rv].
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
    
def emcee_lnprob(theta, x, y, yerr, grid, isonames):
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

    Outputs
    -------
    lnprob : float
         The log probability value, which is the sum of the log prior and the
         log likelihood.

    """
    print(theta)
    lp = emcee_lnprior(theta,grid)
    if not np.isfinite(lp):
        return -np.inf
    return lp + emcee_lnlike(theta, x, y, yerr, grid, isonames)

    

def fit_mcmc(cat,catnames,grid,isonames,caterrnames=None,initpar=None,
             steps=100,extdict=None,cornername=None,verbose=False):
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
    
    # Set up the MCMC sampler
    ndim, nwalkers = 4, 20
    delta = [initpar[0]*0.1, 0.1, 0.1, 0.2]
    pos = [initpar + delta*np.random.randn(ndim) for i in range(nwalkers)]

    x = np.arange(ncat)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, emcee_lnprob,
                                    args=(x,cphot,cphoterr,grid,isonames))

    if cornername is not None: steps=np.maximum(steps,500)  # at least 500 steps
    out = sampler.run_mcmc(pos, steps)

    samples = sampler.chain[:, np.int(steps/2):, :].reshape((-1, ndim))

    # Get the median and stddev values
    pars = np.zeros(ndim,float)
    parerr = np.zeros(ndim,float)
    if verbose is True: print('MCMC values:')
    names = ['age','metal','extinction','distmod']
    for i in range(ndim):
        t=np.percentile(samples[:,i],[16,50,84])
        pars[i] = t[1]
        parerr[i] = (t[2]-t[0])*0.5
    if verbose is True: printpars(pars,parerr)
        
    # The maximum likelihood parameters
    bestind = np.unravel_index(np.argmax(sampler.lnprobability),sampler.lnprobability.shape)
    pars_ml = sampler.chain[bestind[0],bestind[1],:]

    mciso = grid(pars_ml[0],pars_ml[1],pars_ml[2],pars_ml[3])
    isophot = []
    for n in isonames:
        isophot.append(mciso.data[n])
    isophot = np.vstack(tuple(isophot)).T
    mcsumdist,mcmeddist,mcchisq = isocomparison(cphot,isophot,cphoterr)
    
    # Put it into the output structure
    dtype = np.dtype([('pars',float,4),('pars_ml',float,4),('parerr',float,4),
                      ('maxlnlike',float),('chisq',float)])
    out = np.zeros(1,dtype=dtype)
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
        print('Corner plot saved to '+cornername)
        
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
    
    
def fit(cat,catnames,isonames,grid=None,caterrnames=None,
        ages=None,metals=None,extinctions=None,distmod=None,initpar=None,
        fixed=None,extdict=None,cornername=None,figfile=None,verbose=False):
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
    
    # Do a grid search over distance modulues, age, metallicity and extinction
    if initpar is None:
        if verbose: print('Performing grid search')
        bestval,chisq = gridsearch(cat,catnames,grid,isonames,caterrnames=caterrnames,
                                   ages=ages,metals=metals,extinctions=extinctions,
                                   distmod=distmod,fixed=fixed,extdict=extdict)
    else:
        bestval = initpar

    #bestval = [7399999999.999999, -2.0, 0.5, 12.5]
    #chisq = 15578909.947923563
    
    # Run MCMC now
    if verbose: print('Running MCMC')
    mcout,mciso = fit_mcmc(cat,catnames,grid,isonames,caterrnames=caterrnames,
                           initpar=bestval,extdict=extdict,cornername=cornername)

    
    if verbose is True:
        print('Final parameters:')
        printpars(mcout['pars_ml'],mcout['parerr'])
        print('chisq = %5.2f' % mcout['chisq'])
    dtype = np.dtype([('age',np.float32),('ageerr',np.float32),('metal',np.float32),('metalerr',np.float32),
                      ('ext',np.float32),('exterr',np.float32),('distmod',np.float32),('distmoderr',np.float32),
                      ('distance',np.float32),('chisq',np.float32)])
    out = np.zeros(1,dtype=dtype)
    out['age'] = mcout['pars_ml'][0][0]
    out['ageerr'] = mcout['parerr'][0][0]  
    out['metal'] = mcout['pars_ml'][0][1]
    out['metalerr'] =   mcout['parerr'][0][1]  
    out['ext'] = mcout['pars_ml'][0][2]
    out['exterr'] = mcout['parerr'][0][2]  
    out['distmod'] = mcout['pars_ml'][0][3]
    out['distmoderr'] = mcout['parerr'][0][3]  
    out['distance'] = 10**((out['distmod']+5)/5.)/1e3
    out['chisq'] = mcout['chisq']

    #out = Table.read('NGC104_chronos_out.fits')
    #mciso = Table.read('NGC104_mciso.fits')
    #mciso = isochrone.Isochrone(mciso)
    #mciso._ext = out['ext'][0]
    #mciso._distmod = out['distmod'][0]
    
    #figfile = 'NGC104_isofit.png'

    #import pdb; pdb.set_trace()
    
    # Make output plot
    if figfile is not None:
        cmdfigure(figfile,cat,catnames,mciso,isonames,out,verbose=verbose)
    
    return out,mciso
