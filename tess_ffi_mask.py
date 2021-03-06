#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from scipy.signal import fftconvolve

import argparse

# turn off runtime warnings (lots from logic on nans)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 



def size_limit(x,y,image):
    yy,xx = image.shape
    ind = ((y > 0) & (y < yy-1) & (x > 0) & (x < xx-1))
    return ind


def ps1_auto_mask(table,wcs,scale=1):
    """
    Make a source mask using the PS1 catalog
    """
    image = np.zeros(wcs.array_shape)
    r = table.raMean.values
    d = table.decMean.values
    x,y = wcs.all_world2pix(r,d,0)
    x = (x+.5).astype(int)
    y = (y+.5).astype(int)
    m = table.iMeanPSFMag.values
    ind = size_limit(x,y,image)
    x = x[ind]; y = y[ind]; m = m[ind]
    
    maglim = np.zeros_like(image,dtype=float)
    magim = image.copy()
    magim[y,x] = m
    
    
    masks = {}
    
    mags = [[18,17],[17,16],[16,15],[15,14],[14,13.5],[13.5,12]]
    size = (np.array([3,4,5,6,7,8]) * scale).astype(int)
    for i in range(len(mags)):
        m = ((magim > mags[i][1]) & (magim <= mags[i][0])) * 1.
        k = np.ones((size[i],size[i]))
        conv = fftconvolve(m, k,mode='same')#.astype(int)
        masks[str(mags[i][0])] = (conv >.1) * 1.
    masks['all'] = np.zeros_like(image,dtype=float)
    for key in masks:
        masks['all'] += masks[key]
    masks['all'] = (masks['all'] > .1) * 1.
    return masks

def gaia_auto_mask(table,wcs,scale=1):
    """
    Make a source mask from gaia source catalogue
    """
    image = np.zeros(wcs.array_shape)
    r = table.ra.values
    d = table.dec.values
    x,y = wcs.all_world2pix(r,d,0)
    x = (x+.5).astype(int)
    y = (y+.5).astype(int)
    m = table.gaia.values.copy()
    ind = size_limit(x,y,image)
    x = x[ind]; y = y[ind]; m = m[ind]
    
    maglim = np.zeros_like(image,dtype=float)
    magim = image.copy()
    magim[y,x] = m
    
    masks = {}
    
    mags = [[18,17],[17,16],[16,15],[15,14],[14,13.5],[13.5,12],[12,10],[10,9],[9,8],[8,7]]
    size = (np.array([3,4,5,6,7,8,10,14,16,18])*scale).astype(int)
    for i in range(len(mags)):
        m = ((magim > mags[i][1]) & (magim <= mags[i][0])) * 1.
        k = np.ones((size[i],size[i]))
        conv = fftconvolve(m, k,mode='same')#.astype(int)
        masks[str(mags[i][0])] = (conv >.1) * 1.
    masks['all'] = np.zeros_like(image,dtype=float)
    for key in masks:
        masks['all'] += masks[key]
    masks['all'] = (masks['all'] > .1) * 1.
    return masks

    
def Big_sat(table,wcs,scale=1):
    """
    Make crude cross masks for the TESS saturated sources.
    The properties in the mask need some fine tuning.
    """
    image = np.zeros(wcs.array_shape)
    i = (table.gaia.values < 7) #& (gaia.gaia.values > 2)
    sat = table.iloc[i]
    r = sat.ra.values
    d = sat.dec.values
    x,y = wcs.all_world2pix(r,d,0)
    x = x.astype(int)
    y = y.astype(int)
    mags = sat.gaia.values
    ind = size_limit(x,y,image)
    
    x = x[ind]; y = y[ind]; mags = mags[ind]
    
    
    satmasks = []
    for i in range(len(x)):
        mag = mags[i]
        mask = np.zeros_like(image,dtype=float)
        if (mag <= 7) & (mag > 5):
            body   = int(13 * scale)
            length = int(20 * scale)
            width  = int(4 * scale)
        if (mag <= 5) & (mag > 4):
            body   = 15 * scale
            length = int(60 * scale)
            width  = int(10 * scale)
        if (mag <= 4):# & (mag > 4):
            body   = int(25 * scale)
            length = int(115 * scale)
            width  = int(10 * scale)
        body = int(body) # no idea why this is needed, but it apparently is.
        kernal = np.ones((body*2,body*2))
        mask[y[i],x[i]] = 1 
        conv = fftconvolve(mask, kernal,mode='same')#.astype(int)
        mask = (conv >.1) * 1.

        mask[y[i]-length:y[i]+length,x[i]-width:x[i]+width] = 1 
        mask[y[i]-width:y[i]+width,x[i]-length:x[i]+length] = 1 
        
        satmasks += [mask]
    satmasks = np.array(satmasks)
    return satmasks

def Strap_mask(data,size):
    strap_mask = np.zeros_like(data)
    path = '/user/rridden/feet/'
    straps = pd.read_csv(path+'tess_straps.csv')['Column'].values
    strap_mask[:,straps+43] = 1
    big_strap = fftconvolve(strap_mask,np.ones((size,size)),mode='same') > .5
    return big_strap


def Make_bad_pixel_mask(file,image):
    
    data = fits.open(image)[0].data
    header = fits.open(image)[0].header
    bad = np.loadtxt(file,skiprows=3,dtype=object)
    mask = np.zeros_like(data)
    for b in bad:
        b = b.split('(')[-1].split(',')

        x  = int(float(b[0]))
        y  = int(float(b[1]))
        dx = int(float(b[2]))
        dy = int(float(b[3]))

        mask[y:y+dy,x:x+dx] = 1
        
    #Make_fits(mask,name,header)
    mask = mask.astype(int)
    mask = mask *8
    return mask

def Make_fits(data, name, header):
    #print('makefits shape ',data.shape)
    newhdu = fits.PrimaryHDU(data, header = header)
    newhdu.scale('int16', bscale=1.0,bzero=32768.0)
    newhdu.writeto(name,overwrite=True)
    return 

def Make_mask(file,sec,ext,scale,badpix,strapsize):
    path = '/user/rridden/feet/'+str(sec) + '/'
    hdu = fits.open(file)[ext]
    image = hdu.data
    wcs = WCS(hdu)
    cam = str(hdu.header['CAMERA'])
    ccd = str(hdu.header['CCD'])
    ps1 = pd.read_csv(path+'ps1_s' + str(sec)+'_'+cam+ccd+'_footprint.csv')
    gaia = pd.read_csv(path+'gaia_s' + str(sec)+'_'+cam+ccd+'_footprint.csv')

    
    sat = Big_sat(gaia,wcs,scale)
    mg = gaia_auto_mask(gaia,wcs,scale)
    mp = ps1_auto_mask(ps1,wcs,scale)

    sat = (np.nansum(sat,axis=0) > 0).astype(int) * 2 # assign 2 bit 
    mask = ((mg['all']+mp['all']) > 0).astype(int) * 1 # assign 1 bit 
    strap = Strap_mask(image,strapsize).astype(int) * 4 # assign 4 bit 
    if badpix is not None:
        bp = Make_bad_pixel_mask(badpix, file)
        totalmask = mask | sat | strap | bp
    else:
        totalmask = mask | sat | strap
    
    return totalmask

def Update_header(header):
    head = header
    head['STARBIT']  = (1, 'bit value for normal sources')
    head['SATBIT']   = (2, 'bit value for saturated sources')
    head['STRAPBIT'] = (4, 'bit value for straps')
    head['STRAPBIT'] = (8, 'bit value for bad pixels')
    return head

def TESS_source_mask(file,sec,ext, name, badpix, scale, strapsize, sub):
    """
    Make and save a source mask for a TESS image using 
    """
    mask = Make_mask(file,sec,ext,scale,badpix, strapsize)
    
    path = '/user/rridden/feet/'
    hdu = fits.open(file)[ext]
    head = Update_header(hdu.header)
    

    Make_fits(mask,name,head)
    if sub:
        print('Making submasks for straps and bad pixels')
        # make strap submask
        strap = (mask & 4)
        n = name.split('.fits')[0] + '.strap.fits'
        Make_fits(strap, n, head)

        # make bad pixel submask
        bad = (mask & 2) | (mask & 8)
        n = name.split('.fits')[0] + '.badpix.fits'
        Make_fits(bad, n, head)




def define_options(parser=None, usage=None, conflict_handler='resolve'):
    if parser is None:
        parser = argparse.ArgumentParser(usage=usage, conflict_handler=conflict_handler)

    parser.add_argument('-f','--file', default = None, 
            help=('Fits file to make the mask of.'))
    parser.add_argument('-sec','--sector', default = None,
            help=('Sector of data'))
    parser.add_argument('-ext','--extension', default = 0,
            help=('Fits extension of image'))
    parser.add_argument('-o','--output', default = 'default.mask.fits',
            help=('Full output path/name for the created mask'))
    parser.add_argument('-b','--badpix',default = None,
            help=('DS9 region file to mask bad pixels.'))
    parser.add_argument('--scale',default = 1,
            help=('scale factor for the mask size, applies to all masks'))
    parser.add_argument('--strapsize',default = 3,
            help=('size for the strap mask size.'))
    parser.add_argument('--save_submasks',default = False,
            help=('save bad pixel and strap submasks.'))

    return parser

    
if __name__ == '__main__':
    print('Making mask for TESS image')
    parser = define_options()
    args   = parser.parse_args()
    print('got options: ',args)
    file   = args.file
    save   = args.output
    scale  = float(args.scale)
    sub  = args.save_submasks
    strapsize = int(args.strapsize)
    badpix = args.badpix
    ext = int(args.extension)
    sec = args.sector

    TESS_source_mask(file,sec,ext, save, badpix, scale, strapsize, sub)
    print('Made mask for {}, saved as {}'.format(file,save))