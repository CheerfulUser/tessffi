#!/usr/bin/env python
import numpy as np
from scipy.signal import savgol_filter
from copy import deepcopy
from glob import glob
import matplotlib.pyplot as plt
import sigmacut
from scipy.ndimage.filters import convolve
from scipy import interpolate
from astropy.wcs import WCS
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.time import Time
from scipy.ndimage import shift
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import os
import argparse
import tess_bkgsub_class as back


def Save_files(self):
    directory = self.savepath 
    Save_space(directory)
    if self.stacked:
        ref = '_stack_'
    ref = '_' + self.date + '_'
    
    name = directory + sector + ref + str(int(cam) * int(ccd)) + '.pdf'
    figures(self.data,self.background,self.noise,name)

    name = directory + sector + ref + str(int(cam) * int(ccd)) + '.fits.fz'
    Make_fits(self.image,name,self.header)

    name = directory + sector + ref + str(int(cam) * int(ccd)) + '.bkg.fits.fz'
    Make_fits(self.background,name,self.header)

    name = directory + sector + ref + str(int(cam) * int(ccd)) + '.mask.fits.fz'
    Make_fits(self.bitmask,name,self.header)

    name = directory + sector + ref + str(int(cam) * int(ccd)) + '.noise.fits.fz'
    Make_fits(self.noise,name,self.header)
    print('saved: ' + self.sector + ref + str(int(self.camera) * int(self.ccd)))
    return

def Make_fits(data, name, header):

    newhdu = fits.PrimaryHDU(data, header = header)
    newhdu.writeto(name,overwrite=True)
    return 


class TESSref(object):
    def __init__(self):
        #needed
        self.imagefiles = None
        self.noisefiles = None
        # options
        self.mask = None
        self.hduext = 1
        self.smoothing = 12
        self.pedastal = 500
        self.savepath = '.'
        self.method = 'low bkg'
        # calculated
        self.hdu = None
        self.date = None
        self.image = None
        self.background = None
        self.subtracted = None
        self.noise = None
        self.bitmask = None
        self.header = None
        self.wcs = None
        self.saturation = 4.8E4 - self.pedastal
        self.sector = None
        self.camera = None
        self.ccd = None
        self.stacked = True
        self.savename = None

    def define_options(self, parser=None, usage=None, conflict_handler='resolve'):
        if parser is None:
            parser = argparse.ArgumentParser(usage=usage, conflict_handler=conflict_handler)
        parser.add_argument('-i','--input', default=None)
        parser.add_argument('-o','--output',default=None,
                help=('Full save path for main output'),type=str)
        parser.add_argument('-m','--method',default='low bkg',
                help=('Method used to calculate the reference frame: \n'
                    + 'low bkg \n'
                    + 'working on it'),type=str)
        parser.add_argument('-p','--pipeline', default = True, 
                help=('Switch to use pipeline saving function'))
        parser.add_argument('-e','--extension', default = 1,
                help=('extension that the image is stored in. 1 for raw TESS files'))
        parser.add_argument('-s','--smoothing', default = 12,
                help=('Size of the smoothing kernal'))
        parser.add_argument('-off','--offset', default = 500,
                help=('offset to be added to images'))
        return parser

    def low_bkg_ref(self):
        files = self.imagefiles
        summed = np.zeros(len(files)) * np.nan
        ex = self.hduext
        for i in range(len(files)):
            hdu = fits.open(files[i])

            data = hdu[ex].data
            wcs = WCS(hdu[ex].header)
            cut = Cutout2D(data,(1024+44,1024),2048,wcs=wcs)
            data = cut.data 
            wcs = cut.wcs
            data[data <= 0] = np.nan
            if np.nansum(abs(data)) > 0:
                summed[i] = np.nansum(abs(data))

        lim = np.percentile(summed[np.isfinite(summed)],5)
        ind = np.where((summed < lim))[0]
        good = files[ind]
        goods = np.zeros((len(good),2048,2048))
        var = np.zeros((len(good),2048,2048))
        mjd = np.zeros(len(good))
        i = 0
        sat_count = np.zeros_like(data)
        for g in good:
            hdu = fits.open(g)
            data = hdu[ex].data
            wcs = WCS(hdu[ex].header)
            cut = Cutout2D(data,(1024+44,1024),2048,wcs=wcs)
            data = cut.data 
            wcs = cut.wcs
            goods[i] = data 

            e = hdu[2].data
            cut = Cutout2D(e,(1024+44,1024),2048)
            data = cut.data 
            var[i] = data**2  

            jd = hdu[1].header['TSTART'] + hdu[1].header['BJDREFI']
            mjd[i] = Time(jd, format='jd', scale='tdb').mjd
            sat_count[data > 4.8E4 - 500]  += 1
            i += 1
        ref = np.nanmedian(goods,axis=0)
        var = np.nanmedian(var,axis=0)
        hdu[1].header['MJD'] = (np.nanmean(mjd), 'stacked')
        hdu[1].header['NIMAGES'] = (str(len(good)), 'number of images stacked')

        sats = sat_count / len(good) >= (0.05)

        self.hdu = hdu[1]
        self.header = hdu[1].header
        self.image = ref
        self.noise = np.sqrt(var) 
        
        return 




    def Save_space(self,Save):
        """
        Creates a path if it doesn't already exist.
        """
        try:
            if not os.path.exists(Save):
                os.makedirs(Save)
        except FileExistsError:
            pass


    def sigma_mask(self,data,error= None,sigma=3,Verbose= False):
        if type(error) == type(None):
            error = np.zeros(len(data))

        calcaverage = sigmacut.calcaverageclass()
        calcaverage.calcaverage_sigmacutloop(data,Nsigma=sigma
                                             ,median_firstiteration=True,saveused=True)
        if Verbose:
            print("mean:%f (uncertainty:%f)" % (calcaverage.mean,calcaverage.mean_err))
        return calcaverage.clipped

    def Source_mask(self, grid=True):
        """
        doc test
        """
        data = self.image
        if grid:
            data[data<0] = np.nan
            data[data >= np.percentile(data,95)] =np.nan
            grid = np.zeros_like(data)
            size = 32
            for i in range(grid.shape[0]//size):
                for j in range(grid.shape[1]//size):
                    section = data[i*size:(i+1)*size,j*size:(j+1)*size]
                    section = section[np.isfinite(section)]
                    lim = np.percentile(section,1)
                    grid[i*size:(i+1)*size,j*size:(j+1)*size] = lim
            thing = data - grid
        else:
            thing = data
        ind = np.isfinite(thing)
        mask = ((thing <= np.percentile(thing[ind],80,axis=0)) |
               (thing <= np.percentile(thing[ind],10))) * 1.0
        self.mask = mask
        return 


    def Saturation_mask(self):
            saturation = self.image > self.saturation
            self.bitmask[saturation] = self.bitmask[saturation] | (128 | 2)
            print('Saturated pixels masked')
            return 

    def Insert_into_orig(self):
        """
        Insert the cutout image into the original array shape to avoid wcs issues.
        """
        new_image = np.zeros((2078,2136))
        
        b = deepcopy(new_image)
        print(b.shape)
        b[:,:] =  128 | 1
        print(b.shape)
        b[:2048,44:44+2048] = self.bitmask
        print(b.shape)
        self.bitmask = b
        print(self.bitmask.shape)
        data = deepcopy(new_image)
        data[:2048,44:44+2048] = self.subtracted
        self.image = data

        noise = deepcopy(new_image)
        noise[:2048,44:44+2048] = self.noise
        self.noise = noise
        print('Arrays recast into original shape.')
        return 

    def Update_header(self):
        sub = self.subtracted
        sub += self.pedastal # add a pedastal value 
        skysig = np.nanmedian(np.nanstd(sub*convolve(self.mask,np.ones((3,3)))))
        skyadu = np.nanmedian(np.nanmedian(sub*convolve(self.mask,np.ones((3,3)))))
        self.header['NAXIS1'] = 2136
        self.header['NAXIS2'] = 2078
        self.header['SKYADU'] = (skyadu, 'median sky')
        self.header['SKYSIG'] = (skysig, 'median sky noise')
        self.header['NIMAGES'] = (str(int(1)), 'number of images stacked')

        jd = self.header['TSTART'] + self.header['BJDREFI']
        self.header['MJD'] = Time(jd, format='jd', scale='tdb').mjd
        
        if self.background is None:
            newhdu.header['BACKAPP'] = 'T'
        self.header['NOISEIM'] = 1
        self.header['MASKIM'] = 1
        gain = np.nanmean([self.header['GAINA'],self.header['GAINB'],
                            self.header['GAINC'],self.header['GAIND']])
        self.header['GAIN'] = (gain, '[electrons/count] Average CCD output gain')
        self.header['PIXSCALE'] = (21, 'pixel scale in arcsec / pix')
        self.header['SW_PLTSC'] = (21, 'pixel scale in arcsec / pix')
        self.header['PHOTCODE'] = (0x9500, 'photpope index')
        self.header['SATURATE'] = self.saturation
        self.header['STACK'] = self.stacked
        self.header['FLAGBADP'] = (0x1, 'pixel flag for bad pixels')
        self.header['FLAGSAT'] = (0x2, 'pixel flag for saturation')
        self.header['FLAGBKG'] = (0x4, 'pixel flag for bad background')

        print('Header updated')
        return


    def Subtract_background(self):
        bkg = back.TESSbackground()
        bkg.smoothing = self.smoothing
        bkg.image = self.image
        bkg.mask = self.mask
        bkg.Calculate_background()
        self.bitmask = bkg.bitmask
        self.background = bkg.background
        self.subtracted = self.image - self.background
        return 

    def Assign_args(self,args):
        self.imagefiles = args.input
        self.name = args.output
        self.pipeline = args.pipeline
        self.method = args.method
        self.pipeline = args.pipeline
        self.hduext = args.extension
        self.smoothing = args.smoothing
        self.pedastal = args.offset
        return

    def Load_list(self):
        if type(self.imagefiles) == str:
            self.imagefiles = np.loadtxt(self.imagefiles,dtype=object)
        return

    def Background_name(self):
        name = self.savename
        bkgname = name.split('.fits')[0] + '.bkg.fits.fz'
        return bkgname

    def Noise_name(self):
        name = self.savename
        noisename = name.split('.fits')[0] + '.noise.fits.fz'
        return noisename

    def Bitmask_name(self):
        name = self.savename
        maskname = name.split('.fits')[0] + '.mask.fits.fz'
        return maskname


    def Pipeline_save(self):
        if self.savename is None:
            self.savename = './not_named_ref.fits.fz'

        name = self.savename
        Make_fits(self.subtracted,name,self.header)

        name = self.Background_name()
        Make_fits(self.background,name,self.header)

        name = self.Noise_name()
        Make_fits(self.noise,name,self.header)

        name = self.Bitmask_name()
        Make_fits(self.bitmask,name,self.header)
        print('All files saved')
        return


    def Make_reference(self,args):
        self.Assign_args(args)
        self.Load_list()
        self.low_bkg_ref()
        self.Source_mask()
        self.Subtract_background()
        self.Saturation_mask()
        self.Insert_into_orig()
        print(self.image.shape)
        self.Update_header()
        if self.pipeline:
            self.Pipeline_save()
        else:
            Save_files(self)
    
        print('Done!')
        return


if __name__ == '__main__':
    print('starting!!!!!')
    ref = TESSref()
    parser = ref.define_options()
    args = parser.parse_args()
    print('got options',args)
    ref.Make_reference(args)

    print('Made reference image')
































