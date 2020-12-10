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


# turn off runtime warnings (lots from logic on nans)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 


def Save_space(Save):
    """
    Creates a path if it doesn't already exist.
    """
    try:
        if not os.path.exists(Save):
            os.makedirs(Save)
    except FileExistsError:
        pass


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def sigma_mask(data,error= None,sigma=3,Verbose= False):
    """
    Create a sigma clip mask for a given array
    """
    if type(error) == type(None):
        error = np.zeros(len(data))

    calcaverage = sigmacut.calcaverageclass()
    calcaverage.calcaverage_sigmacutloop(data,Nsigma=sigma
                                         ,median_firstiteration=True,saveused=True)
    if Verbose:
        print("mean:%f (uncertainty:%f)" % (calcaverage.mean,calcaverage.mean_err))
    return calcaverage.clipped

def Source_mask(data, grid=True):
    """
    Create a source mask for a given image using sigma clipping. If grid is True 
    then a limited background subtraction is performed on a 32x32 pix grid to
    get a better idea of sources with a basic background subtraction.
    """
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

    return mask 

def figures(data, bkg, err, save):

    plt.figure(figsize=(8,8))
    plt.subplot(2,2,1)
    plt.title('Raw')
    im = plt.imshow(data,origin='',vmin=np.percentile(data,10),
                vmax=np.percentile(data,90))
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.subplot(2,2,2)
    plt.title('Error')
    im = plt.imshow(err,origin='',vmin=np.percentile(err,10),
                vmax=np.percentile(err,90))
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.subplot(2,2,3)
    plt.title('Background')
    im = plt.imshow(bkg,origin='',vmin=np.percentile(bkg,10),
                vmax=np.percentile(bkg,90))
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)


    plt.subplot(2,2,4)
    sub = data - bkg
    plt.title('Subbed')
    im = plt.imshow(sub,origin='',vmin=np.percentile(sub,10),
                vmax=np.percentile(sub,90))
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.tight_layout()
    plt.savefig(save)
    plt.close()
    return

def Saturation_mask(self):
        saturation = self.image > self.saturation
        self.bitmask[saturation] = self.bitmask[saturation] | (128 | 2)
        print('Saturated pixels masked')
        return self



def Save_files(self):
    """
    Creates the save paths for the output files.
    """
    if self.savepath is None:
        self.savepath = '.'
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




class TESSbackground():
    """
    Class for calculating the background of tess images.
    """
    def __init__(self):
        #needed
        self.image = None
        # options
        self.mask = None
        self.smoothing = 12
        self.strap = True
        # calculated
        self.background = None
        self.noise = None
        self.bitmask = None


    def Smooth_bkg(self,data,smoothing_factor, extrapolate = True):
        d = deepcopy(data)
        d[d == 0] = np.nan
        x = np.arange(0, d.shape[1])
        y = np.arange(0, d.shape[0])
        arr = np.ma.masked_invalid(d)
        xx, yy = np.meshgrid(x, y)
        #get only the valid values
        x1 = xx[~arr.mask]
        y1 = yy[~arr.mask]
        newarr = arr[~arr.mask]

        estimate = interpolate.griddata((x1, y1), newarr.ravel(),
                                  (xx, yy),method='linear')
        bitmask = np.zeros_like(data,dtype=int)
        bitmask[np.isnan(estimate)] = 128 | 4
        nearest = interpolate.griddata((x1, y1), newarr.ravel(),
                                  (xx, yy),method='nearest')
        if extrapolate:
            estimate[np.isnan(estimate)] = nearest[np.isnan(estimate)]

        estimate = gaussian_filter(estimate,smoothing_factor)

        return estimate, bitmask

    def Strap_bkg(self,data):

        ind = np.where(np.nansum(abs(self.image),axis=0)>0)[0]
        strap_bkg = np.zeros_like(data)
        for col in ind:
            x = np.arange(0,data.shape[1])
            y = data[:,col].copy()
            finite = np.isfinite(y)
            if len(y[finite]) > 5:
                finite = np.isfinite(y)
                bad = sigma_mask(y[finite],sigma=2)
                finite = np.where(finite)[0]
                y[finite[bad]] = np.nan
                finite = np.isfinite(y)
                #regressionLine = np.polyfit(x[finite], y[finite], 3)
                fit = UnivariateSpline(x[finite], y[finite])
                fit.set_smoothing_factor(1500)
                #p = interp1d(x[finite], y[finite],bounds_error=False,fill_value=np.nan,kind='cubic')
                #p = np.poly1d(regressionLine)
                p = fit(x)
                finite = np.isfinite(p)
                smooth =savgol_filter(p[finite],13,3)
                p[finite] = smooth

                thingo = y - p
                finite = np.isfinite(thingo)
                bad = sigma_mask(thingo[finite],sigma=2)
                finite = np.where(finite)[0]
                y[finite[bad]] = np.nan
                finite = np.isfinite(y)
                #regressionLine = np.polyfit(x[finite], y[finite], 3)
                #p = np.poly1d(regressionLine)
                #p = interp1d(x[finite], y[finite],bounds_error=False,fill_value=np.nan,kind='cubic')
                fit = UnivariateSpline(x[finite], y[finite])
                fit.set_smoothing_factor(1500)
                p = fit(x)
                finite = np.isfinite(p)
                smooth =savgol_filter(p[finite],13,3)
                p[finite] = smooth
                strap_bkg[:,col] = p

        return strap_bkg

    def Calculate_background(self):
        if self.mask is None:
            print('No source mask is given, calculatng one from this image')
            self.mask = Source_mask(self.image)

        mask = deepcopy(self.mask)
        data = deepcopy(self.image)

        strap_mask = np.zeros_like(data)
        straps = pd.read_csv('tess_straps.csv')['Column'].values
        strap_mask[:,straps-1] = 1
        big_strap = convolve(strap_mask,np.ones((3,3))) > 0
        big_mask = convolve((mask==0)*1,np.ones((8,8))) > 0

        masked = deepcopy(data) * ((big_mask==0)*1) * ((big_strap==0)*1)
        masked[masked == 0] = np.nan
        
        bkg_smooth, bitmask = self.Smooth_bkg(masked,self.smoothing)
        round1 = data - bkg_smooth
        if self.strap:
            print('calculating strap background')
            round2 = round1 * ((big_strap==1)*1) * ((big_mask==1)*1)
            round2[round2 == 0] = np.nan
            strap_bkg = self.Strap_bkg(round2)
        else:
            print('Not calculating strap background')
            strap_bkg = 0

        self.background = strap_bkg + bkg_smooth
        self.bitmask = bitmask
        print('Background calculated')
        return 

def Make_fits(data, name, header):
    #print('makefits shape ',data.shape)
    newhdu = fits.PrimaryHDU(data, header = header)
    newhdu.writeto(name,overwrite=True)
    return 

class TESS_reduction(object):
    """
    Performs background subtraction on TESS FFIs and formats them for photpipe.

    Inputs
    ------

    """
    def __init__(self):
        #needed
        self.file = None
        self.reffile = None
        # options
        self.savename = None
        self.pipeline = True
        self.plot = False
        self.smoothing = 12
        self.pedastal = 500
        self.strap = True
        self.savepath = '.'
        # calculated
        self.mask = None
        self.hdu = None
        self.date = None
        self.image = None
        self.reference = None
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
        self.stacked = False




    def define_options(self, parser=None, usage=None, conflict_handler='resolve'):
        if parser is None:
            parser = argparse.ArgumentParser(usage=usage, conflict_handler=conflict_handler)

        parser.add_argument('-i','--image', default = None, 
                help=('Image to reduce'))
        parser.add_argument('-r','--reference', default = None,
                help=('reference image'))
        parser.add_argument('-o','--output',default=None,
                help=('Full save path for main output'),type=str)
        parser.add_argument('-p','--pipeline', type=str2bool, nargs='?',
                        const=True, default=True, 
                help=('Switch to use pipeline saving function'))
        parser.add_argument('-strap','--strap', type=str2bool, nargs='?',
                        const=True, default=True,
                help=('Include the strap background subtraction'))
        parser.add_argument('-s','--smoothing', default = 12,
                help=('Size of the smoothing kernal'))
        parser.add_argument('-off','--offset', default = 500,
                help=('offset to be added to images'))
        parser.add_argument('-fig','--figure',type=str2bool, nargs='?',
                        const=True, default=False,
                help=('Switch to plot reduction figures'))
        return parser


    def Save_space(Save):
        """
        Creates a path if it doesn't already exist.
        """
        try:
            if not os.path.exists(Save):
                os.makedirs(Save)
        except FileExistsError:
            pass


    

    def Source_mask(self, grid=True):
        """
        Create a source mask for a given image using sigma clipping. If grid is True 
        then a limited background subtraction is performed on a 32x32 pix grid to
        get a better idea of sources with a basic background subtraction.
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
                    if len(section) > 10:
                        lim = np.percentile(section,1)
                    else:
                        lim = np.nan
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
        """
        Find all saturated pixels and assign them in the bitmask
        """
        if self.bitmask is None:
            self.bitmask = np.zeros_like(self.image,dtype=int)
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
        b[:,:] =  128 | 1
        b[:2048,44:44+2048] = self.bitmask
        self.bitmask = b

        data = deepcopy(new_image)
        data[:2048,44:44+2048] = self.subtracted
        self.image = data

        data = deepcopy(new_image)
        data[:2048,44:44+2048] = self.background
        self.background = data

        noise = deepcopy(new_image)
        noise[:2048,44:44+2048] = self.noise
        self.noise = noise
        print('Arrays recast into original shape.')
        return 

    def Update_header(self):
        """
        Update the fits header to the format needed for photpipe
        """
        sub = self.subtracted
        sub += self.pedastal # add a pedastal value 
        skysig = np.nanmedian(np.nanstd(sub*convolve(self.mask,np.ones((3,3)))))
        skyadu = np.nanmedian(np.nanmedian(sub*convolve(self.mask,np.ones((3,3)))))
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


    def Load_image(self):
        if self.file is None:
            raise ValueError('No file specified')
        try:
            hdu = fits.open(self.file)
        except:
            raise ValueError('Could not load {}'.format(self.file))

        self.date = self.file.split('tess')[-1].split('-')[0]
        self.sector = self.file.split('-')[1]
        self.camera = self.file.split('-')[2]
        self.ccd = self.file.split('-')[3]

        self.hdu = hdu[1]
        self.header = hdu[1].header

        data = hdu[1].data
        err = hdu[2].data 
        self.wcs = WCS(hdu[1].header)
        cut = Cutout2D(data,(1024+44,1024),2048,wcs=self.wcs)
        self.image = cut.data
        wcs = cut.wcs
        err = Cutout2D(err,(1024+44,1024),2048).data
        self.noise = err
        print('Successfully loaded {}'.format(self.file))
        return 
    
    def Load_reference(self):
        """
        Load in reference file
        """
        if self.reffile is None:
            raise ValueError('No reference file specified')
        if os.path.isfile(self.reffile):
            hdu = fits.open(self.reffile)
        else:
            raise ValueError('{} Does not exist. Create the file with tess_reference.py'.format(self.reffile))        

        data = hdu[0].data
        cut = Cutout2D(data,(1024+44,1024),2048)
        self.reference = cut.data
        return

    def Subtract_background(self):
        """
        Use the TESSbackground class to subtract the background
        """
        bkg = TESSbackground()
        bkg.image = self.image
        bkg.mask = self.mask
        bkg.strap = self.strap
        bkg.smoothing = self.smoothing
        bkg.Calculate_background()
        self.background = bkg.background
        self.subtracted = self.image - self.background
        return


    def Background_name(self):
        """
        Background image file name
        """
        name = self.savename
        bkgname = name.split('.fits')[0] + '.bkg.fits.fz'
        return bkgname

    def Noise_name(self):
        """
        Noise image file name
        """
        name = self.savename
        noisename = name.split('.fits')[0] + '.noise.fits.fz'
        return noisename

    def Bitmask_name(self):
        """
        Bitmask image file name
        """
        name = self.savename
        maskname = name.split('.fits')[0] + '.mask.fits.fz'
        return maskname


    def Pipeline_save(self):
        """
        Save pipeline outputs
        """
        if self.savename is None:
            self.savename = './not_named_reduction.fits.fz'

        name = self.savename
        Make_fits(self.image,name,self.header)

        name = self.Background_name()
        Make_fits(self.background,name,self.header)

        name = self.Noise_name()
        Make_fits(self.noise,name,self.header)

        name = self.Bitmask_name()
        Make_fits(self.bitmask,name,self.header)
        print('All files saved')
        return


    def Assign_args(self,args):
        self.file = args.image
        self.reffile = args.reference
        self.savename = args.output
        self.pipeline = args.pipeline
        self.smoothing = args.smoothing
        self.strap = args.strap
        self.pedastal = args.offset
        self.plot = args.figure
        return

    def Run_reduction(self,args):
        """"
        Wrapper for all the reduction functions.
        """
        self.Assign_args(args)
        # load images 
        self.Load_image()
        if np.nansum(self.image) > 100:
            self.Load_reference()
            # get the source mask from the reference image
            self.Source_mask()
            # get background and subtract from image
            self.Subtract_background()
            # identify saturated pixels
            self.Saturation_mask()
            # update header with info
            self.Update_header()
            # apply offset
            self.image += self.pedastal
            # put back into the original format
            self.Insert_into_orig()

            if self.pipeline:
                self.Pipeline_save()
            else:
                Save_files(self)
    
        print('Done!')
        return



if __name__ == '__main__':
    print('starting')
    tess = TESS_reduction()
    parser = tess.define_options()
    args = parser.parse_args()
    print('got options: ',args)
    tess.Run_reduction(args)

    print('Subtracted background')




