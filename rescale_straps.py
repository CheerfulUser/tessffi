#!/usr/bin/env python

import numpy as np
from copy import deepcopy
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from astropy.stats import sigma_clip
from astropy.io import fits
import argparse


def grad_clip(data,box_size=100):
    gradind = np.zeros_like(data)
    
    for i in range(len(data)):
        if i < box_size//2:
            d = data[:i+box_size//2]
        elif len(data) - i < box_size//2:
            d = data[i-box_size//2:]
        else:
            d = data[i-box_size//2:i+box_size//2]
        
        ind = np.isfinite(d)
        d = d[ind]
        if len(d) > 5:
            gind = ~sigma_clip(np.gradient(abs(d))+d,sigma=2).mask

            if i < box_size//2:
                gradind[:i+box_size//2][ind] = gind
            elif len(data) - i < box_size//2:
                gradind[i-box_size//2:][ind] = gind
            else:
                gradind[i-box_size//2:i+box_size//2][ind] = gind
        
    return gradind > 0

def fit_strap(data):
    
    x = np.arange(0,len(data))
    y = data.copy()
    y[~grad_clip(y)] = np.nan
    finite = np.isfinite(y)
    if len(y[finite]) > 5:
        finite = np.isfinite(y)
        #y = median_clipping(y)
        finite = np.where(finite)[0]
        finite = np.isfinite(y)
        y[finite] = savgol_filter(y[finite],31,3)
        p = interp1d(x[finite], y[finite],bounds_error=False,fill_value=np.nan,kind='linear')
        p = p(x)
        p[np.isfinite(p)] = savgol_filter(p[np.isfinite(p)],31,1)
    return p


def Make_fits(data, name, header):
    #print('makefits shape ',data.shape)
    newhdu = fits.PrimaryHDU(data, header = header)
    newhdu.scale('int16', bscale=1.0,bzero=32768.0)
    newhdu.writeto(name,overwrite=True)
    return 

def Rescale_straps(ref_file,mask_file,output,av_size=5):
    ref = fits.open(ref_file)[0]
    data = ref.data * 1.
    mask = fits.open(mask_file)[0].data

    sind = np.where(np.nansum((mask & 4),axis=0)>0)[0]
    normals = np.where(np.nansum((mask & 4),axis=0)==0)[0]
    
    QE = np.zeros_like(mask) * 1.
    corrected = deepcopy(data)
    for i in sind:
        closest = np.nanmin(abs(normals-i))
        nind = normals[abs(normals-i) < closest+av_size]
        s1 = fit_strap(data[:,i])
        n1 = fit_strap(np.nanmedian(data[:,nind],axis=1))
        factor = np.nanmedian(s1/n1)
        QE[:,i] = factor
        corrected[:,i] =  corrected[:,i] / factor
    
    cor_header = deepcopy(ref.header)
    cor_header['STRAPCAL'] = ('T', 'T if the strap QE has been rescaled')

    Make_fits(corrected, output, cor_header)
    name = output.split('.fits')[0] + '.qe.fits'
    Make_fits(QE*100, name, cor_header) # scaled to make it work in integer form 

def define_options(parser=None, usage=None, conflict_handler='resolve'):
    if parser is None:
        parser = argparse.ArgumentParser(usage=usage, conflict_handler=conflict_handler)

    parser.add_argument('-f','--ref_file', default = None, 
            help=('Reference fits file to rescale strap QE.'))
    parser.add_argument('-m','--mask', default = None,
            help=('Full pipeline mask for the image.'))
    parser.add_argument('-o','--output', default = 'default.rescale.fits',
            help=('Full output path/name for the rescaled image'))
    parser.add_argument('--av_size',default = 5,
            help=('number of nearby columns to average.'))

    return parser

    
if __name__ == '__main__':
    print('Rescaling strap QE for TESS image')
    parser  = define_options()
    args    = parser.parse_args()
    print('got options: ',args)
    file    = args.ref_file
    mask    = args.mask
    save    = args.output
    av_size = int(args.av_size)

    Rescale_straps(file, mask, save, av_size)
    print('Made mask for {}, saved as {}'.format(file,save))
