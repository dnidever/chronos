#!/usr/bin/env python

"""ISOGRID.PY - Isochrone grid class

"""

__authors__ = 'David Nidever <dnidever@montana.edu?'
__version__ = '20210920'  # yyyymmdd

import os
import time
import numpy as np
from glob import glob
from astropy.wcs import WCS
from astropy.io import fits
from astropy.table import Table
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d

def isointerp2(iso1,iso2,frac,photnames=None):
    """ Interpolate between two isochrones."""
    # frac: fractional distance for output isochrone
    #  0 is close to iso1 and 1 is close to iso2

    niso1 = len(iso1)
    niso2 = len(iso2)
    label1 = np.unique(iso1['LABEL'])
    label2 = np.unique(iso2['LABEL'])    
    age1 = iso1['AGE'][0]
    metal1 = iso1['METAL'][0]
    age2 = iso2['AGE'][0]
    metal2 = iso2['METAL'][0]    
    
    # Use MINI, original mass

    # get unique MINI values between the two isochrones
    #mini = np.concatenate((iso1['MINI'].data,iso2['MINI']))
    #mini = np.unique(mini)
        
    # Maybe interpolate different star type "labels" separately
    # 1-9
    # 1: main sequence
    # 2: subgiant branch (Hertzsprung gap)
    # 3: red giant branch
    # 4: horizontal branch
    # 5:  sometimes "missing" 
    # 6:  sometimes "missing"
    # 7: AGB
    # 8: TAGB
    # 9: WD (often only one point)

    # Descriptions from CMD website
    #   http://stev.oapd.inaf.it/cmd_3.5/faq.html
    # 0 = PMS, pre main sequence
    # 1 = MS, main sequence
    # 2 = SGB, subgiant branch, or Hertzsprung gap for more intermediate+massive stars
    # 3 = RGB, red giant branch, or the quick stage of red giant for intermediate+massive stars
    # 4 = CHEB, core He-burning for low mass stars, or the very initial stage of CHeB for intermediate+massive stars
    # 5 = still CHEB, the blueward part of the Cepheid loop of intermediate+massive stars
    # 6 = still CHEB, the redward part of the Cepheid loop of intermediate+massive stars
    # 7 = EAGB, the early asymptotic giant branch, or a quick stage of red giant for massive stars
    # 8 = TPAGB, the thermally pulsing asymptotic giant branch
    # 9 = post-AGB (in preparation!)
    #     
    # can do 1-3 together
    # sometimes 5+6 are "missing"
    # 

    # if a phase is missing in ONE of the isochrones then drop it from the output one as well

    # for isochrones of the SAME age, you should be able to use "mass" as the independent variable
    # to interpolate things

    # the points line up REALLY well on the MINI vs. INT_IMF plot

    # get unique MINI values between the two isochrones
    #mini = np.concatenate((iso1['MINI'].data,iso2['MINI']))
    #mini = np.unique(mini)

    # MINI values for isochrones of the same age are quite similar, sometimes one will have a couple extra points

    if photnames is None:
        colnames = np.char.array(iso1.colnames)
        photind, = np.where((colnames.find('MAG')>-1) & (colnames.find('_')>-1))
        photnames = list(colnames[photind])
    interpnames = ['INT_IMF','MASS']+photnames

    # Initialize the output catalog
    nout = np.max([niso1,niso2])
    out = Table()
    out['AGE'] = np.zeros(nout,float)+age1
    out['METAL'] = metal1
    out['MINI'] = 0.0
    out['INT_IMF'] = 0.0
    out['MASS'] = 0.0        
    out['LABEL'] = 0
    for n in photnames:
        out[n] = 0.0
    
    # Label loop
    count = 0
    for l in np.arange(1,9):
        #print(l)
        lab1 = iso1['LABEL']==l
        nlab1 = np.sum(lab1)
        lab2 = iso2['LABEL']==l
        nlab2 = np.sum(lab2)        
        # both must have this label
        if nlab1==0 or nlab2==0:
            continue
        # match up the Mini values, they should be VERY similar
        # just use the one with the fewer number of points
        mini1 = iso1['MINI'][lab1].data
        # need to scale by the maximum mini
        mini1 /= np.max(mini1)
        mini2 = iso2['MINI'][lab2].data
        mini2 /= np.max(mini2)
        if nlab1<nlab2:
            mini = mini1
        else:
            mini = mini2
        # make sure we are in the range for both MINI arrays
        gd, = np.where( (mini>=np.min(mini1)) & (mini<=np.max(mini1)) &
                        (mini>=np.min(mini2)) & (mini<=np.max(mini2)))
        mini = mini[gd]
        nmini = len(mini)
        maxmini = np.max(iso1['MINI'][lab1].data)*(1-frac)+np.max(iso2['MINI'][lab2].data)*frac
            
        # Interpolate each using the mini
        out['MINI'][count:count+nmini] = mini*maxmini
        for n in interpnames:
            data1 = interp1d(mini1,iso1[n][lab1],kind='quadratic')(mini)
            data2 = interp1d(mini2,iso2[n][lab2],kind='quadratic')(mini)
            # use linear interpolation to the value at FRAC
            data = data1*(1-frac)+data2*frac
            out[n][count:count+nmini] = data
        out['LABEL'][count:count+nmini] = l
        count += nmini
        
    # Trim extra elements
    out = out[out['LABEL']>0]

    return out
    
    
def isointerp(grid,age,metal,names=None):
    """ Interpolate isochrones."""
    # Input isochrone grid and desired age/metal

    # Get closest neighbors
    # Ages
    #  check for exact match
    if age in grid.ages:
        loaind, = np.where(grid.ages==age)
        hiaind = None
        loage = grid.ages[loaind]
    else:
        loaind, = np.where(grid.ages <= age)
        hiaind, = np.where(grid.ages > age)
        nhiaind = len(hiaind)
        if nhiaind>0:
            loaind = np.max(loaind)
            hiaind = np.min(hiaind)
        if nhiaind==0:  # at upper edge
            loaind, = np.where(grid.ages < age)
            loaind = np.min(loaind)
            hiaind, = np.where(grid.ages >= age)
            hiaind = np.max(hiaind)
        loage = grid.ages[loaind]
        hiage = grid.ages[hiaind]
    # Metals
    #  check for exact match
    if metal in grid.ages:
        lomind, = np.where(grid.metals==metal)
        himind = None
        lometal = grid.metals[lomind]
    else:
        lomind, = np.where(grid.metals <= metal)
        himind, = np.where(grid.metals > metal)
        nhimind = len(himind)
        if nhiaind>0:
            lomind = np.max(lomind)
            himind = np.min(himind)
        if nhimind==0:  # at upper edge
            lomind, = np.where(grid.metals < metal)
            lomind = np.min(lomind)
            himind, = np.where(grid.metals >= metal)
            himind = np.max(himind)
        lometal = grid.metals[lomind]
        himetal = grid.metals[himind]            
            
    # Now do the interpolation
    if hiaind is None:
        nages = 1
    else:
        nages = 2
    if himind is None:
        nmetals = 1
    else:
        nmetals = 2

    #iso1 = grid._data[grid._index[grid._ind2ind1[loaind,lomind]]]
    #iso2 = grid._data[grid._index[grid._ind2ind1[loaind,himind]]]
    #frac = (metal-lometal)/(himetal-lometal)
    #isoloa = isointerp2(iso1,iso2,frac)  

    
    # Interpolate in age first
    #-------------------------
    if nages==2:
        # low metallicity        
        iso1 = grid._data[grid._index[grid._ind2ind1[loaind,lomind]]]
        iso2 = grid._data[grid._index[grid._ind2ind1[hiaind,lomind]]]
        frac = (age-loage)/(hiage-loage)
        isolom = isointerp2(iso1,iso2,frac,names)
        # high metallicity        
        iso1 = grid._data[grid._index[grid._ind2ind1[loaind,himind]]]
        iso2 = grid._data[grid._index[grid._ind2ind1[hiaind,himind]]]
        frac = (age-loage)/(hiage-loage)
        isohim = isointerp2(iso1,iso2,frac,names)
    else:
        isolom = grid._data[grid._index[grid._ind2ind1[loaind,lomind]]]
        isohim = grid._data[grid._index[grid._ind2ind1[loaind,himind]]]

    # Interpolate in metallicity
    #---------------------------
    frac = (metal-lometal)/(himetal-lometal)
    iso = isointerp2(isolom,isohim,frac,names)
    
    return iso

    
class IsoGrid:

    def __init__(self,iso):
        uages = np.unique(iso['AGE'].data)
        self._ages = uages
        self._agerange = [np.min(uages),np.max(uages)]
        umetals = np.unique(iso['METAL'].data)
        self._metals = umetals
        self._metalrange = [np.min(umetals),np.max(umetals)]
        self._index = []
        self._data = iso

        # Photometric bands
        colnames = np.char.array(iso.colnames)
        photind, = np.where((colnames.find('MAG')>-1) & (colnames.find('_')>-1))
        photnames = list(colnames[photind])
        self.bands = photnames
        
        # Create the index
        index = []
        allages = []
        allmetals = []
        npoints = []
        ind2ind1 = np.zeros((len(uages),len(umetals)),int)
        count = 0
        for i,a in enumerate(uages):
            for j,m in enumerate(umetals):
                ind, = np.where((iso['AGE']==a) & (iso['METAL']==m))
                index.append(ind)
                allages.append(a)
                allmetals.append(m)
                npoints.append(len(ind))
                ind2ind1[i,j] = count
                count += 1
        self._index = index
        self._allages = np.array(allages)
        self._allmetals = np.array(allmetals)
        self._npoints = np.array(npoints)
        self._ind2ind1 = ind2ind1

    def __repr__(self):
        """ Print out string representation."""
        s = repr(self.__class__)+'\n'
        s += '%d models [%d ages, %d metals]\n' % (len(self._allages),len(self.ages),len(self.metals))
        s += '%6.3e < Age < %6.3e years\n' % (self.minage,self.maxage)
        s += '%6.2f < Metal < %6.2f\n' % (self.minmetal,self.maxmetal)
        s += str(len(self.bands))+' bands: '+', '.join(self.bands)
        return s
        
    def __call__(self,age,metal,names=None):
        """ Return the isochrone for this age and metallicity."""

        # Check that the requested values are inside our grid
        if age<self.agerange[0] or age>self.agerange[1] or metal<self.metalrange[0] or metal>self.metalrange[1]:
            raise ValueError('age=%6.3e metal=%6.2f is outside the isochrone grid. %6.3e<age<%6.3e, %6.2f<metal<%6.2f' %
                             (age,metal,self.agerange[0],self.agerange[1],self.metalrange[0],self.metalrange[1]))
    
        # Exact match exists
        if age in self.ages and metal in self.metals:
            aind, = np.where(self.ages==age)
            mind, = np.where(self.metals==metal)
            ind1 = self._ind2ind1[aind,mind]
            index = self._index[ind1]
            iso = self._data[index]

            # Only return the columns that we want
            if names is None:
                names = self.bands
            outnames = ['INT_IMF','MASS']+names

            # Initialize the output catalog
            niso = len(iso)
            out = Table()
            out['AGE'] = np.zeros(niso,float)
            out['METAL'] = 0.0
            out['MINI'] = 0.0
            out['INT_IMF'] = 0.0
            out['MASS'] = 0.0        
            out['LABEL'] = 0
            for n in photnames:
                out[n] = 0.0
            for n in outnames:
                out[n] = iso[n]
            
            return out
        
        # Need to interpolate
        else:
            return self.interp(age,metal,names=names)
        
    @property
    def ages(self):
        """ Return unique ages."""
        return self._ages

    @property
    def agerange(self):
        """ Return the range of unique ages."""
        return self._agerange

    @property
    def minage(self):
        """ Return the minimum age."""
        return self._agerange[0]

    @property
    def maxage(self):
        """ Return the maximum age."""
        return self._agerange[1]
    
    @property
    def metals(self):
        """ Return unique metals."""
        return self._metals

    @property
    def metalrange(self):
        """ Return the range of unique metallicity."""
        return self._metalrange

    @property
    def minmetal(self):
        """ Return the minimum metallicity."""
        return self._metalrange[0]

    @property
    def maxmetal(self):
        """ Return the maximum metallicity."""
        return self._metalrange[1]
    
    def interp(self,age,metal,names=None):
        """ Interpolation grid for this age and metallicity."""
        return isointerp(self,age,metal,names=names)


