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
    mini = np.concatenate((iso1['MINI'].data,iso2['MINI']))
    mini = np.unique(mini)
        
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
    mini = np.concatenate((iso1['MINI'].data,iso2['MINI']))
    mini = np.unique(mini)

    # MINI values for isochrones of the same age are quite similar, sometimes one will have a couple extra points

    if photnames is None:
        colnames = np.char.array(iso1.colnames)
        photind, = np.where((colnames.find('MAG')>-1) & (colnames.find('_')>-1))
        photnames = list(colnames[photind])
    interpnames = ['INT_IMF']+photnames

    # Initialize the output catalog
    nout = np.max([niso1,niso2])
    out = Table()
    out['AGE'] = np.zeros(nout,float)+age1
    out['METAL'] = metal1
    out['MINI'] = 0.0
    out['INT_IMF'] = 0.0
    out['LABEL'] = 0
    for n in photnames:
        out[n] = 0.0
    
    # Label loop
    count = 0
    for l in np.arange(1,9):
        print(l)
        lab1 = iso1['LABEL']==l
        nlab1 = np.sum(lab1)
        lab2 = iso2['LABEL']==l
        nlab2 = np.sum(lab2)        
        # both must have this label
        if nlab1==0 or nlab2==0:
            continue
        # same age
        if age1==age2:
            # match up the Mini values, they should be VERY similar
            # just use the one with the fewer number of points
            mini1 = iso1['MINI'][lab1].data
            #X1 = np.vstack((mini1,np.zeros(nlab1,float))).T
            mini2 = iso2['MINI'][lab2].data
            #X2 = np.vstack((mini2,np.zeros(nlab2,float))).T
            #if nlab1<nlab2:
            #    kdt = cKDTree(X2)
            #    dist,ind = kdt.query(X1,k=1,p=2)
            #    #diff = np.diff(mini2)
            #    #diff = np.hstack((diff,diff[-1]))
            #else:
            #    kdt = cKDTree(X1)
            #    dist,ind = kdt.query(X2,k=1,p=2)
            if nlab1<nlab2:
                mini = mini1
            else:
                mini = mini2
            # make sure we are in the range for both MINI arrays
            gd, = np.where( (mini>=np.min(mini1)) & (mini<=np.max(mini1)) &
                            (mini>=np.min(mini2)) & (mini<=np.max(mini2)))
            mini = mini[gd]
            nmini = len(mini)
                
            # Interpolate each using the mini
            import pdb; pdb.set_trace()
            out['MINI'][count:count+nmini] = mini
            for n in interpnames:
                data1 = interp1d(mini1,iso1[n][lab1],kind='quadratic')(mini)
                data2 = interp1d(mini2,iso2[n][lab2],kind='quadratic')(mini)
                # use linear interpolation to the value at FRAC
                data = data1*(1-frac)+data2*frac
                out[n][count:count+nmini] = data
            out['LABEL'][count:count+nmini] = l
            count += nmini
            
        # different age
        else:
            import pdb; pdb.set_trace()
        
        # Trim extra elements
        out = out[out['LABEL']>0]
    
    import pdb; pdb.set_trace()

    return out
    
    
def isointerp(isogrid,age,metal,names=None):
    """ Interpolate isochrones."""
    # Input isochrone grid and desired age/metal

    # Get closest neighbors
    # Ages
    #  check for exact match
    if age in isogrid.ages:
        loaind, = np.where(isogrid.ages==age)
        hiaind = None
        loage = isogrid.ages[loaind]
    else:
        loaind, = np.where(isogrid.ages <= age)
        hiaind, = np.where(isogrid.ages > age)
        nhiaind = len(hiaind)
        if nhiaind>0:
            loaind = np.max(loaind)
            hiaind = np.min(hiaind)
        if nhiaind==0:  # at upper edge
            loaind, = np.where(isogrid.ages < age)
            loaind = np.min(loaind)
            hiaind, = np.where(isogrid.ages >= age)
            hiaind = np.max(hiaind)
        loage = isogrid.ages[loaind]
        hiage = isogrid.ages[hiaind]
    # Metals
    #  check for exact match
    if metal in isogrid.ages:
        lomind, = np.where(isogrid.metals==metal)
        himind = None
        lometal = isogrid.metals[lomind]
    else:
        lomind, = np.where(isogrid.metals <= metal)
        himind, = np.where(isogrid.metals > metal)
        nhimind = len(himind)
        if nhiaind>0:
            lomind = np.max(lomind)
            himind = np.min(himind)
        if nhimind==0:  # at upper edge
            lomind, = np.where(isogrid.metals < metal)
            lomind = np.min(lomind)
            himind, = np.where(isogrid.metals >= metal)
            himind = np.max(himind)
        lometal = isogrid.metals[lomind]
        himetal = isogrid.metals[himind]            
            
    # Now do the interpolation
    if hiaind is None:
        nages = 1
    else:
        nages = 2
    if himind is None:
        nmetals = 1
    else:
        nmetals = 2

    #ages = np.zeros(niso,float)
    #metals = np.zeros(niso,float)    
    #if hiaind is not None:
    #    iso1 = isogrid._data[isogrid._index[isogrid._ind2ind1[loaind,lomind]]]
    #    iso2 = isogrid._data[isogrid._index[isogrid._ind2ind1[hiaind,lomind]]]
    #    ages[[0,1]] = [isogrid.ages[loaind],isogrid.ages[hiaind]]
    #    metals[[0,1]] = [isogrid.metals[lomind],isogrid.metals[lomind]]
    #if himind is not None:
    #    iso3 = isogrid._data[isogrid._index[isogrid._ind2ind1[loaind,himind]]]
    #    iso4 = isogrid._data[isogrid._index[isogrid._ind2ind1[hiaind,himind]]]        
    #    ages[[2,3]] = [isogrid.ages[loaind],isogrid.ages[hiaind]]
    #    metals[[2,3]] = [isogrid.metals[himind],isogrid.metals[himind]]

    iso1 = isogrid._data[isogrid._index[isogrid._ind2ind1[loaind,lomind]]]
    iso2 = isogrid._data[isogrid._index[isogrid._ind2ind1[loaind,himind]]]
    frac = (metal-lometal)/(himetal-lometal)
    isoloa = isointerp2(iso1,iso2,frac)  

    
    # Interpolate in age first
    #-------------------------
    if nages==2:
        # low metallicity        
        iso1 = isogrid._data[isogrid._index[isogrid._ind2ind1[loaind,lomind]]]
        iso2 = isogrid._data[isogrid._index[isogrid._ind2ind1[hiaind,lomind]]]
        frac = (age-loage)/(hiage-loage)
        isolom = isointerp2(iso1,iso2,frac,names)
        # high metallicity        
        iso1 = isogrid._data[isogrid._index[isogrid._ind2ind1[loaind,himind]]]
        iso2 = isogrid._data[isogrid._index[isogrid._ind2ind1[hiaind,himind]]]
        frac = (age-loage)/(hiage-loage)
        isohim = isointerp2(iso1,iso2,frac,names)


    # Interpolate in metallicity
    #---------------------------

    
    import pdb; pdb.set_trace()

    
class IsoGrid:

    def __init__(self,iso):
        uages = np.unique(iso['AGE'])
        self._ages = uages
        self._agerange = [np.min(uages),np.max(uages)]
        umetals = np.unique(iso['METAL'])
        self._metals = umetals
        self._metalrange = [np.min(umetals),np.max(umetals)]
        self._index = []
        self._data = iso

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
        
    def __call__(self,age,metal):
        """ Return the isochrone for this age and metallicity."""

        # Check that the requested values are inside our grid
        if age<self.agerange[0] or age>self.agerange[1] or metal<self.metalrange[0] or metal>self.metalrange[1]:
            raise ValueError('age=%6.3e metal=%6.2f is outside the isochrone grid. %6.3e<age<%6.3e, %6.2f<metal<%6.2f' %
                             (age,metal,self.agerange[0],self.agerange[1],self.metalrange[0],self.metalrange[1]))
    
        # Exact match exists
        if age in self.ages and metal in self.metals:
            aind, = np.where(self.ages==age)
            mind, = np.where(self.metals==metal)
            ind1 = self.ind2ind1[aind,mind]
            index = self._index[ind1]
            iso = self._data[index]
            return iso
        # Need to interpolate
        else:
            return self.interp(age,metal)
        
    @property
    def ages(self):
        """ Return unique ages."""
        return self._ages

    @property
    def agerange(self):
        """ Return the range of unique ages."""
        return self._agerange
    
    @property
    def metals(self):
        """ Return unique metals."""
        return self._metals

    @property
    def metalrange(self):
        """ Return the range of unique metallicity."""
        return self._metalrange

    def interp(self,age,metal):
        """ Interpolation grid for this age and metallicity."""
        return isointerp(self,age,metal)


