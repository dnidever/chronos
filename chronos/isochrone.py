#!/usr/bin/env python

"""ISOCHRONE.PY - Isochrone and isochrone grid classes

"""

__authors__ = 'David Nidever <dnidever@montana.edu?'
__version__ = '20210920'  # yyyymmdd

import os
import numpy as np
from glob import glob
from astropy.table import Table
from astropy.io import fits
from scipy.interpolate import interp1d
from dlnpyutils import utils as dln
import copy
from . import extinction,utils


def load(files=None):
    """ Load all the default isochrone files."""
    if files is None:
        ddir = utils.datadir()
        files = glob(ddir+'parsec_grid.fits.gz')
        files.sort()
        nfiles = len(files)
        if nfiles==0:
            raise Exception("No default isochrone files found in "+ddir)
    else:
        files = [files]
    iso = []
    for f in files:
        iso.append(Table.read(f))
    if len(iso)==1: iso=iso[0]
    
    # Index
    grid = IsoGrid(iso)
    
    return grid

def isointerp2(iso1,iso2,frac,photnames=None,minlabel=1,maxlabel=7,verbose=False):
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

    isominimax1 = np.max(iso1['MINI'])
    isominimax2 = np.max(iso2['MINI'])
    intimfmax1 = np.max(iso1['INT_IMF'])
    intimfmax2 = np.max(iso2['INT_IMF'])    

    # max MINI for the output isochrone
    isominimax = isominimax1*(1-frac)+isominimax2*frac
    
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
    interpnames = ['INT_IMF','MASS','LOGTE','LOGG']+photnames

    # Initialize the output catalog
    nout = int(1.5*np.max([niso1,niso2]))
    out = Table()
    out['AGE'] = np.zeros(nout,float)+(age1*(1-frac)+age2*frac)
    out['METAL'] = metal1*(1-frac)+metal2*frac
    out['MINI'] = 0.0
    out['INT_IMF'] = 0.0
    out['MASS'] = 0.0
    out['LOGTE'] = 0.0
    out['LOGG'] = 0.0            
    out['LABEL'] = -1
    for n in photnames:
        out[n] = 0.0
    
    # Label loop
    count = 0
    for l in np.arange(minlabel,maxlabel+1):
        lab1 = iso1['LABEL']==l
        nlab1 = np.sum(lab1)
        lab2 = iso2['LABEL']==l
        nlab2 = np.sum(lab2)
        # both must have this label
        if nlab1==0 or nlab2==0:
            continue
        if verbose:
            print('Label=%d, N1=%d, N2=%d' % (l,nlab1,nlab2))
        # Multiple data points to interpolate
        if nlab1>1 and nlab2>1:
            # match up the Mini values, they should be VERY similar
            # just use the one with the fewer number of points
            mini1 = iso1['MINI'][lab1].data
            mini1 /= isominimax1        
            mini1min = np.min(mini1)
            mini1max = np.max(mini1)
            xmini1 = (mini1-mini1min)/(mini1max-mini1min)  # scale MINI to 0 and 1
            mini2 = iso2['MINI'][lab2].data
            mini2 /= isominimax2        
            mini2min = np.min(mini2)
            mini2max = np.max(mini2)
            xmini2 = (mini2-mini2min)/(mini2max-mini2min)  # scale MINI to 0 and 1
            # need to scale by the maximum mini

            # Use the XMINI values with the largest number of points
            if nlab1>nlab2:
                xmini = xmini1
                dointerp1 = False
            else:
                xmini = xmini2
                dointerp1 = True
            nxmini = len(xmini)
            

            #if nlab1<nlab2:
            #    mini = mini1
            #else:
            #    mini = mini2
            ## make sure we are in the range for both MINI arrays
            #gd, = np.where( (mini>=np.min(mini1)) & (mini<=np.max(mini1)) &
            #                (mini>=np.min(mini2)) & (mini<=np.max(mini2)))
            #mini = mini[gd]
            #nmini = len(mini)


            # Interpolate the min and max "scaled" MINI value for the requested isochrone
            # then take the mini array with the largest number of points and "scale" this
            # (from min/max value) for all three sets

            # min/max MINI values for output isochrone
            minimin = mini1min*(1-frac)+mini2min*frac
            minimax = mini1max*(1-frac)+mini2max*frac
            mini = xmini*(minimax-minimin)+minimin
            # now scale up by "global" isochrone MINI maximum
            mini *= isominimax
        
            # Interpolate each using the mini
            #out['MINI'][count:count+nmini] = mini*maxmini
            out['MINI'][count:count+nxmini] = mini
            for n in interpnames:
                kind = 'quadratic'
                #if nlab1<3 or nlab2<3: kind='linear'
                #data1 = interp1d(mini1,iso1[n][lab1],kind=kind)(mini)
                #data2 = interp1d(mini2,iso2[n][lab2],kind=kind)(mini)
                if nlab1<3 or nlab2<3: kind='linear'
                if dointerp1:
                    data1 = interp1d(xmini1,iso1[n][lab1],kind=kind)(xmini)
                    data2 = iso2[n][lab2]
                else:
                    data1 = iso1[n][lab1]
                    data2 = interp1d(xmini2,iso2[n][lab2],kind=kind)(xmini)
                        
                # use linear interpolation to the value at FRAC
                data = data1*(1-frac)+data2*frac
                out[n][count:count+nxmini] = data
            out['LABEL'][count:count+nxmini] = l
            count += nxmini


        # Single data point, often for label=9, WD
        else:
            uselab1, = np.where(lab1)
            uselab2, = np.where(lab2)
            # take middle point
            if len(uselab1)==1 and len(uselab2)>1:
                uselab2 = uselab2[len(uselab2)//2]
            elif len(uselab2)==1 and len(uselab1)>1:
                uselab1 = uselab1[len(uselab1)//2]
            out['MINI'][count] = iso1['MINI'][uselab1]*(1-frac)+iso2['MINI'][uselab2]*frac
            for n in interpnames:
                out[n][count] = iso1[n][uselab1]*(1-frac)+iso2[n][uselab2]*frac
            out['LABEL'][count] = l
            count += 1
        
    # Trim extra elements
    out = out[out['LABEL']>-1]
    
    return out
    
    
def isointerp(grid,age,metal,names=None,minlabel=1,maxlabel=7,verbose=False):
    """ Interpolate isochrones."""
    # Input isochrone grid and desired age/metal

    # Get closest neighbors
    # Ages
    #  check for exact match
    if age in grid.ages:
        loaind = dln.first_el(np.where(grid.ages==age)[0])
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
        lomind = dln.first_el(np.where(grid.metals==metal)[0])
        himind = None
        lometal = grid.metals[lomind]
    else:
        lomind, = np.where(grid.metals <= metal)
        himind, = np.where(grid.metals > metal)
        nhimind = len(himind)
        if nhimind>0:
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
        
    # Interpolate in age first
    #-------------------------
    # two ages
    if nages==2:
        # Two metallicities
        if himind is not None:
            # low metallicity
            if verbose:
                print('Interpolating between ages for low metallicity (%6.3f): %6.2e and %6.2e' % (lometal,loage,hiage))
            iso1 = grid._data[grid._index[grid._ind2ind1[loaind,lomind]]]
            iso2 = grid._data[grid._index[grid._ind2ind1[hiaind,lomind]]]
            frac = (age-loage)/(hiage-loage)
            isolom = isointerp2(iso1,iso2,frac,names,minlabel=minlabel,maxlabel=maxlabel,verbose=verbose)
            # high metallicity
            if verbose:
                print('Interpolating between ages for high metallicity (%6.3f): %6.2e and %6.2e' % (himetal,loage,hiage))        
            iso1 = grid._data[grid._index[grid._ind2ind1[loaind,himind]]]
            iso2 = grid._data[grid._index[grid._ind2ind1[hiaind,himind]]]
            frac = (age-loage)/(hiage-loage)
            isohim = isointerp2(iso1,iso2,frac,names,minlabel=minlabel,maxlabel=maxlabel,verbose=verbose)
        # Single metallicity
        else:
            if verbose:
                print('Interpolating between ages for metallicity (%6.3f): %6.2e and %6.2e' % (lometal,loage,hiage))
            iso1 = grid._data[grid._index[grid._ind2ind1[loaind,lomind]]]
            iso2 = grid._data[grid._index[grid._ind2ind1[hiaind,lomind]]]
            frac = (age-loage)/(hiage-loage)
            isolom = isointerp2(iso1,iso2,frac,names,minlabel=minlabel,maxlabel=maxlabel,verbose=verbose)
    # Single age, two metallicities
    else:
        isolom = grid._data[grid._index[grid._ind2ind1[loaind,lomind]]]
        isohim = grid._data[grid._index[grid._ind2ind1[loaind,himind]]]
            
    # Interpolate in metallicity
    #---------------------------
    if himind is not None:
        if verbose:
            print('Interpolating between metallicites for age (%6.2e): %6.3f and %6.3f' % (age,lometal,himetal))
        frac = (metal-lometal)/(himetal-lometal)
        iso = isointerp2(isolom,isohim,frac,names,minlabel=minlabel,maxlabel=maxlabel,verbose=verbose)
    else:
        iso = isolom

    # Return an Isochrone object
    iso = Isochrone(iso)
    
    return iso


def isopdf(data):
    # PDF, probability distribution function
    # int_IMF, which is the integral of the IMF under consideration (as selected in the form, in number of stars,
    # and normalised to a total mass of 1 Msun) from 0 up to the current M_ini. Differences between 2 values of
    # int_IMF give the absolute number of stars occupying that isochrone section per unit mass of stellar
    # population initially born, as expected for the selected IMF.
    ndata = len(data)
    pdf = np.maximum(np.diff(data['INT_IMF'].data),0.0)   # 1e-8
    pdf = np.hstack((0.0, pdf))
    # Create normalized index array (from 0-1)
    indx = np.arange(ndata).astype(float)/(ndata-1)
    # Create cumulative distribution function
    cdf = np.cumsum(pdf)
    cdf /= np.max(cdf)
    return pdf,cdf
    
def smoothsynth(iso1,iso2,totmass,photnames=None,verbose=False):
    """ Create synthetic stellar population that smoothly covers
        between two isochrones """

    minlabel = np.min(np.concatenate((iso1['LABEL'],iso2['LABEL'])))
    maxlabel = np.max(np.concatenate((iso1['LABEL'],iso2['LABEL'])))
    
    age1 = iso1['AGE'][0]
    metal1 = iso1['METAL'][0]
    age2 = iso2['AGE'][0]
    metal2 = iso2['METAL'][0]
    dt = iso1.data.dtype

    # Total mass input, figure out the number of stars we expect
    # for our stellar mass range
    nstars = np.ceil((np.max(iso1['INT_IMF'])-np.min(iso1['INT_IMF']))*totmass)

    if photnames is None:
        colnames = np.char.array(iso1.colnames)
        photind, = np.where((colnames.find('MAG')>-1) & (colnames.find('_')>-1))
        photnames = list(colnames[photind])
    interpnames = ['AGE','METAL','MINI','INT_IMF','MASS','LOGTE','LOGG']+photnames

    # Label loop
    pdf1,cdf1 = isopdf(iso1)
    outlist = []
    for l in range(minlabel,maxlabel+1):
        lab1 ,= np.where(iso1['LABEL']==l)
        nlab1 = len(lab1)
        lab2, = np.where(iso2['LABEL']==l)
        nlab2 = len(lab2)
        # both must have this label
        if nlab1==0 or nlab2==0:
            continue
        if verbose:
            print('Label=%d, N1=%d, N2=%d' % (l,nlab1,nlab2))

        # Only one point, get next or previous point
        if l==maxlabel:
            lab1 = np.hstack((lab1[0]-1,lab1))
            lab2 = np.hstack((lab2[0]-1,lab2))
        else:
            lab1 = np.hstack((lab1[0],lab1,lab1[-1]))
            lab2 = np.hstack((lab2[0],lab2,lab2[-1]))
        nlab1 = len(lab1)
        nlab2 = len(lab2)

        # Number of stars for this label
        nstars_label = int(np.sum(pdf1[lab1])*totmass)
        frac = np.random.rand(nstars_label)
            
        # Get scaled indexes
        pdf_label = pdf1[lab1]
        pdf_label = np.hstack((0.0, pdf_label))
        pindx = np.arange(len(pdf_label)).astype(float)/len(pdf_label)
        cdf_label = np.cumsum(pdf_label)
        cdf_label /= np.max(cdf_label)
        rnd = np.random.rand(nstars_label)
        newindx = interp1d(cdf_label,pindx)(rnd)

        # Normalized indices for iso1 and iso2
        indx1 = np.arange(nlab1).astype(float)/(nlab1-1)
        indx2 = np.arange(nlab2).astype(float)/(nlab2-1)
            
        out1 = np.zeros(nstars_label,dtype=dt)
        for n in interpnames:
            kind = 'quadratic'
            if nlab1<3 or nlab2<3: kind='linear'
            data1 = interp1d(indx1,iso1[n][lab1],kind=kind)(newindx)
            data2 = interp1d(indx2,iso2[n][lab2],kind=kind)(newindx)             
            # use linear interpolation to the value at FRAC
            data = data1*(1-frac)+data2*frac
            out1[n] = data
        out1['LABEL'] = l
        outlist.append(out1)

    # Stack all of the label tables
    out = np.hstack(outlist)
    out = Table(out)

    return out


# Maybe make separate Iso class for the single isochrones

class Isochrone:
    def __init__(self,iso,extdict=None):
        self._age = np.min(iso['AGE'].data)
        self._metal = np.min(iso['METAL'].data)
        self._distmod = 0.0
        self._ext = 0.0
        
        self._data = copy.deepcopy(iso)
        # make sure the data are properly sorted by MINI
        self._data.sort('MINI')
        self.ndata = len(iso)
        self._origdata = iso
        
        # Photometric bands
        colnames = np.char.array(iso.colnames)
        photind, = np.where((colnames.find('MAG')>-1) & (colnames.find('_')>-1))
        photnames = list(colnames[photind])
        self._bands = photnames

        # Extinction dictionary
        if extdict is not None:
            self._extdict = extdict
        else:
            extdict,exttab = extinction.load()
            self._extdict = extdict
            self._exttab = exttab
        
    def __call__(self,distmod=None,ext=None,maxlabel=None):
        """ Return an isochrone with a given distance modulus and extinction."""
        out = self.copy()
        # Label limit
        if maxlabel is not None:
            gd, = np.where(out.data['LABEL']<=maxlabel)
            out._data = out.data[gd]
            out.ndata = len(gd)
            
        # Add distance modulus
        if distmod is not None:
            for n in self.bands:
                out[n] += distmod
        # Add extinction
        if ext is not None:
            pass
        
        return out

    def __repr__(self):
        """ String representation."""
        out = self.__class__.__name__+'\n'
        out += 'Age = %8.3e years\n' % self.age
        out += 'Metallicity = %6.3f\n' % self.metal
        out += 'Distance Modulus = %6.3f\n' % self.distmod
        out += 'Extinction = %6.3f\n' % self.ext
        out += 'Nbands = %d' % len(self.bands)
        return out 

    def __len__(self):
        return len(self._data)

    @property
    def size(self):
        return len(self._data)

    @property
    def shape(self):
        return (len(self._data),)
    
    def __setitem__(self,index,data):
        self._data[index] = data
    
    def __getitem__(self,index):
        return self._data[index]
    
    @property
    def data(self):
        """ Return the data."""
        return self._data
    
    @property
    def distmod(self):
        """ Return the distance modulus."""
        return self._distmod

    @distmod.setter
    def distmod(self,distm):
        """ Set the distance modulus."""
        if distm != self.distmod:
            diffdistmod = distm-self._distmod
            for n in self.bands:
                self._data[n] += diffdistmod
            self._distmod = distm
            
    @property
    def distance(self):
        """ Return the distance in kpc."""
        return 10**(self.distmod*0.2+1.0)/1e3

    @property
    def ext(self):
        """ Return the A(V) extinction."""
        return self._ext

    @ext.setter
    def ext(self,extin):
        """ Set the extinction."""
        newdata = self._origdata.copy()
        # Extinct it
        newdata = extinction.extinct(newdata,extin,extdict=self._extdict,isonames=self.bands)
        self._data = newdata
        self._data.sort('MINI')  # make sure they are sorted right
        # Set the distance modulus
        distm = copy.deepcopy(self._distmod)
        self._distmod = 0.0
        self.distmod = distm
        self._ext = extin
        
    @property
    def extinction(self):
        """ Return the A(V) extinction.  Convenience function."""
        return self.ext 
    @property
    def age(self):
        """ Return the age."""
        return self._age

    @property
    def metal(self):
        """ Return the metallicity."""
        return self._metal

    @property
    def colnames(self):
        """ Return the isochrone data column names."""
        return self.data.colnames
    
    @property
    def bands(self):
        """ Return the list of bands."""
        return self._bands

    def synth(self,nstars=None,totmass=None,minlabel=1,maxlabel=8,bands=None,minmass=0,
              maxmass=1000,columns=['AGE','METAL','MINI','MASS','LOGTE','LOGG','LABEL']):
        """ Create synthetic population."""
    
        if bands is None:
            bands = self.bands

        # By default use 1000 stars
        if nstars is None and totmass is None:
            nstars = 1000

        lab = ((self._data['LABEL']>=minlabel) & (self._data['LABEL']<=maxlabel))
        data = self._data[lab]

        ## Mass cuts
        massind, = np.where((data['MINI'] >= minmass) & (data['MINI'] <= maxmass))
        data = data[massind]
        ndata = len(data)
        if ndata==0:
            raise ValueError('No isochrone points left after mass cut')

        # Total mass input, figure out the number of stars we expect
        # for our stellar mass range
        if nstars is None and totmass is not None:
            nstars = np.ceil((np.max(data['INT_IMF'])-np.min(data['INT_IMF']))*totmass)
            
        # Initialize the output catalog
        out = Table()
        out['AGE'] = np.zeros(int(nstars),float)
        for c in columns:
            if c != 'AGE':
                out[c] = 0.0
        for n in bands:  # bands to interpolate
            out[n] = 0.0        

        # PDF, probability distribution function
        # int_IMF, which is the integral of the IMF under consideration (as selected in the form, in number of stars,
        # and normalised to a total mass of 1 Msun) from 0 up to the current M_ini. Differences between 2 values of
        # int_IMF give the absolute number of stars occupying that isochrone section per unit mass of stellar
        # population initially born, as expected for the selected IMF.
        pdf = np.maximum(np.diff(data['INT_IMF'].data),0.0)   # 1e-8
        pdf = np.hstack((0.0, pdf))
        
        # Create normalized index array (from 0-1)
        indx = np.arange(ndata).astype(float)/(ndata-1)
        # Create cumulative distribution function
        cdf = np.cumsum(pdf)
        cdf /= np.max(cdf)

        # Get the indices in the structure from the CDF
        #interp,cdf,indx,randomu(seed,nstars),newindx
        newindx = interp1d(cdf,indx)(np.random.rand(int(nstars)))
        
        # Interpolate all of the relevant columns
        for n in out.colnames:
            if n != 'INT_IMF':
                newval = interp1d(indx,data[n])(newindx)
                out[n] = newval

        return out
            
    @classmethod
    def read(cls,filename):
        """ Read in an isochrone file."""
        hdu = fits.open(filename)
        head = hdu[0].header
        data = Table(hdu[1].data)
        extdicttable = hdu[2].data
        extdict = {}
        for n in extdicttable.names:
            extdict[n] = extdicttable[n][0]
        age = head['AGE']
        metal = head['METAL']
        ext = head['EXT']
        distmod = head['DISTMOD']
        iso = Isochrone(data,extdict=extdict)
        iso.ext = ext
        iso.distmod = distmod
        return iso
        
    def write(self,filename):
        """ Write an isochrone to a file."""
        if os.path.exists(filename): os.remove(filename)
        hdu = fits.HDUList()
        hdu.append(fits.table_to_hdu(self._data))
        hdu[0].header['AGE'] = self.age
        hdu[0].header['METAL'] = self.metal
        hdu[0].header['EXT'] = self.ext
        hdu[0].header['DISTMOD'] = self.distmod
        extdict = list(self._extdict.items())
        extdicttable = Table()
        key,val = extdict[0]
        extdicttable[key] = np.zeros(1,float)+val
        for i in np.arange(1,len(extdict)):
            key,val = extdict[i]
            extdicttable[key] = val
        hdu.append(fits.table_to_hdu(extdicttable))
        hdu.writeto(filename,overwrite=True)
        hdu.close()
        
    def copy(self):
        """ Return a copy of self."""
        return copy.deepcopy(self)
    
    
class IsoGrid:

    def __init__(self,iso,extdict=None):
        # Change metallicity and age names for parsec
        if 'AGE' not in iso.colnames and 'LOGAGE' in iso.colnames:
            iso['AGE'] = 10**iso['LOGAGE'].copy()
        if 'METAL' not in iso.colnames and 'MH' in iso.colnames:
            iso['METAL'] = iso['MH']
        uages = np.unique(iso['AGE'].data)
        self._ages = uages
        self._nages = len(uages)
        self._agerange = [np.min(uages),np.max(uages)]
        umetals = np.unique(iso['METAL'].data)
        self._metals = umetals
        self._nmetals = len(umetals)
        self._metalrange = [np.min(umetals),np.max(umetals)]
        self._index = []
        self._data = iso

        # Photometric bands
        colnames = np.char.array(iso.colnames)
        photind, = np.where((colnames.find('MAG')>-1) & (colnames.find('_')>-1))
        photnames = list(colnames[photind])
        self._bands = photnames
        
        # Create the index
        index = []
        allages = []
        allmetals = []
        npoints = []
        ind2ind1 = np.zeros((len(uages),len(umetals)),int)
        count = 0
        ageindex = dln.create_index(iso['AGE'])
        for i,a in enumerate(uages):
            aind = ageindex['index'][ageindex['lo'][i]:ageindex['hi'][i]+1]
            mindex = dln.create_index(iso[aind]['METAL'])
            for j,m in enumerate(mindex['value']):
                mind = mindex['index'][mindex['lo'][j]:mindex['hi'][j]+1]
                ind = aind[mind]
                ind = ind[np.argsort(ind)]  # MAKE SURE they are SORTED!
                #ind, = np.where((iso['AGE']==a) & (iso['METAL']==m))
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
        
        # Extinction dictionary
        if extdict is not None:
            self._extdict = extdict
        else:
            extdict,exttab = extinction.load()
            self._extdict = extdict
            self._exttab = exttab
            
        
    def __repr__(self):
        """ Print out string representation."""
        s = repr(self.__class__)+'\n'
        s += '%d models [%d ages, %d metals]\n' % (len(self._allages),len(self.ages),len(self.metals))
        s += '%6.3e < Age < %6.3e years\n' % (self.minage,self.maxage)
        s += '%6.2f < Metal < %6.2f\n' % (self.minmetal,self.maxmetal)
        s += str(len(self.bands))+' bands: '+', '.join(self.bands)
        return s

    def __len__(self):
        """ Return the number of isochrones in the grid."""
        return self._nages*self._nmetals

    @property
    def size(self):
        """ Return the number of isochrones in the grid."""
        return self._nages*self._nmetals
    
    @property
    def shape(self):
        """ Return the grid shape."""
        return (self._nages,self._nmetals)

    def __getitem__(self, item):
        """ Get single isochrones with indexing or slicing."""

        # Single slice or integer
        #   make sure we have values for each dimension        
        if type(item) is not tuple:
            item = (item,0)
        if type(item[0]) is not int or type(item[1]) is not int:
            raise ValueError('Currently only single indexing is supported.  No slicing.')

        if item[0]>self._nages-1:
            raise IndexError('index '+str(item[0])+' is out of bounds for axis 0 with size '+str(self._nages))
        if item[1]>self._nmetals-1:
            raise IndexError('index '+str(item[1])+' is out of bounds for axis 1 with size '+str(self._nmetals))
        
        age = self.ages[item[0]]
        metal = self.metals[item[1]]
        return self(age,metal)
            
    def __call__(self,age,metal,ext=None,distmod=None,names=None,system=None,
                 closest=False,verbose=False):
        """ Return the isochrone for this age and metallicity."""

        # Check that the requested values are inside our grid
        if age<self.agerange[0] or age>self.agerange[1] or metal<self.metalrange[0] or metal>self.metalrange[1]:
            raise ValueError('age=%6.3e metal=%6.2f is outside the isochrone grid. %6.3e<age<%6.3e, %6.2f<metal<%6.2f' %
                             (age,metal,self.agerange[0],self.agerange[1],self.metalrange[0],self.metalrange[1]))
            
        # The columns that we want
        if names is None and system is not None:
            nameind, = np.where(np.char.array(self.bands).find(system.upper())>-1)
            if len(nameind)==0:
                raise ValueError('Not photometric bands for system='+system)
            names = list(np.array(self.bands)[nameind])
        if names is None:
            names = self.bands
        outnames = ['AGE','METAL','MINI','INT_IMF','MASS','LOGTE','LOGG','LABEL']+names
        
        
        # Get the closest isochrone on the grid
        if closest:
            nei = self.neighbors(age,metal)
            neiages = []
            neimetals = []
            for a,m in nei:
                neiages.append(a)
                neimetals.append(m)
            neiages = np.array(neiages)
            neimetals = np.array(neimetals)
            dist = np.sqrt((np.log10(neiages)-np.log10(age))**2+(neimetals-metal)**2)
            bestind = np.argmin(dist)
            bestage = neiages[bestind]
            bestmetal = neimetals[bestind]
            if verbose:
                print('Closest grid point is age=%6.2e, metal=%6.3f' % (bestage,bestmetal))

            aind, = np.where(self.ages==bestage)
            mind, = np.where(self.metals==bestmetal)
            ind1 = self._ind2ind1[aind[0],mind[0]]
            index = self._index[ind1]
            iso = self._data[index]

            # Initialize the output catalog
            niso = len(iso)
            out = Table()
            out['AGE'] = np.zeros(niso,float)
            for n in outnames:
                out[n] = iso[n]

            # Create Isochrone object
            outiso = Isochrone(out,extdict=self._extdict)
                
            # Add distance modulus and extinction
            if distmod is not None:
                outiso.distmod = distmod
            if ext is not None:
                outiso.ext = ext

            return outiso
        
        # Exact match exists
        #   basically within the rounding error
        #if (np.min(np.abs(self.ages-age)) < 1e5 and
        #    np.min(np.abs(self.metals-metal)) < 1e-3):
        #    aind = np.argmin(np.abs(self.ages-age))
        #    mind = np.argmin(np.abs(self.metals-metal))
        #    ind1 = self._ind2ind1[aind,mind]
        if age in self.ages and metal in self.metals:
            aind, = np.where(self.ages==age)
            mind, = np.where(self.metals==metal)
            ind1 = self._ind2ind1[aind[0],mind[0]]
            index = self._index[ind1]
            iso = self._data[index]

            if verbose:
                print('Exact match for age=%6.2e, metal=%6.3f' % (age,metal))

            # Initialize the output catalog
            niso = len(iso)
            out = Table()
            out['AGE'] = np.zeros(niso,float)
            for n in outnames:
                out[n] = iso[n]

            # Create Isochrone object
            outiso = Isochrone(out,extdict=self._extdict) 
        
        # Need to interpolate
        else:
            if verbose:
                print('Interpolating')
            outiso = self.interp(age,metal,names=names,verbose=verbose)
            outiso._extdict = self._extdict
            
        # Add distance modulus and extinction
        if distmod is not None:
            outiso.distmod = distmod
        if ext is not None:
            outiso.ext = ext
                
        return outiso

    @property
    def bands(self):
        """ Return the list of bands."""
        return self._bands
    
    @property
    def colnames(self):
        """ Return the isochrone data column names."""
        return self._data.colnames    
            
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

    def neighbors(self,age,metal):
        """ Get the closest neighbors below and above this point in the grid
            needed for interpolation."""

        # Get closest neighbors
        # Ages
        #  check for exact match
        if age in self.ages:
            loaind, = np.where(self.ages==age)
            hiaind = None
            loage = self.ages[loaind[0]]
        else:
            loaind, = np.where(self.ages <= age)
            hiaind, = np.where(self.ages > age)
            nhiaind = len(hiaind)
            if nhiaind>0:
                loaind = np.max(loaind)
                hiaind = np.min(hiaind)
            if nhiaind==0:  # at upper edge
                loaind, = np.where(self.ages < age)
                loaind = np.min(loaind)
                hiaind, = np.where(self.ages >= age)
                hiaind = np.max(hiaind)
            loage = self.ages[loaind]
            hiage = self.ages[hiaind]
        # Metals
        #  check for exact match
        if metal in self.ages:
            lomind, = np.where(self.metals==metal)
            himind = None
            lometal = self.metals[lomind[0]]
        else:
            lomind, = np.where(self.metals <= metal)
            himind, = np.where(self.metals > metal)
            nhimind = len(himind)
            if nhimind>0:
                lomind = np.max(lomind)
                himind = np.min(himind)
            if nhimind==0:  # at upper edge
                lomind, = np.where(self.metals < metal)
                lomind = np.min(lomind)
                himind, = np.where(self.metals >= metal)
                himind = np.max(himind)
            lometal = self.metals[lomind]
            himetal = self.metals[himind]   

        if hiaind is None and himind is None:
            return ([loage,lometal])
        if hiaind is None:
            return ([loage,lometal],[loage,himetal])
        if himind is None:
            return ([loage,lometal],[hiage,lometal])
        return ([loage,lometal],[loage,himetal],[hiage,lometal],[hiage,himetal])
        
    def interp(self,age,metal,names=None,minlabel=1,maxlabel=7,verbose=False):
        """ Interpolate in grid for this age and metallicity."""
        return isointerp(self,age,metal,names=names,minlabel=minlabel,maxlabel=maxlabel,verbose=verbose)
    
    def copy(self):
        """ Return a copy of self."""
        return copy.deepcopy(self)

    @classmethod
    def read(cls,files):
        """ Read from files """
        return load(files)
    
    #def write(self):
    #  write to file and keep index as well
