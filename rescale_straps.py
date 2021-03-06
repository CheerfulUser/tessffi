#!/usr/bin/env python

import numpy as np
from copy import deepcopy
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from astropy.stats import sigma_clip
from astropy.io import fits
import argparse
import multiprocessing
from joblib import Parallel, delayed

# turn off runtime warnings (lots from logic on nans)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

def grad_clip(data,box_size=100):
    """
    Perform a local sigma clip of points based on the gradient of the points. 
    Pixels with large gradients are contaminated by stars/galaxies.

    Inputs
    ------
        data : array
            1d array of the data to clip
        box_size : int 
            integer defining the box size to clip over 
    Output
    ------
        gradind : bool

    """
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
    
    gradind = gradind > 0
    return gradind 

def fit_strap(data):
    """
    interpolate over missing data

    """
    x = np.arange(0,len(data))
    y = data.copy()
    y[~grad_clip(y)] = np.nan
    finite = np.isfinite(y)
    p =np.ones_like(x) * np.nan
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


def Make_fits(data, name, header,integer = True):
    #print('makefits shape ',data.shape)
    newhdu = fits.PrimaryHDU(data, header = header)
    if integer:
        newhdu.scale('int16', bscale=1.0,bzero=32768.0)
    else:
        newhdu.scale('float64')
    newhdu.writeto(name,overwrite=True)
    return 

def Rescale_straps(ref_file,mask_file,output,av_size=5):
    """
    Rescales the strap columns of the input TESS image to account 
    for the increased QE of strap columns.

    Inuts
    -----
        ref_file : str
            file to be rescaled 
        mask_file : str
            master mask file for the image 
        output : str
            save name
        av_size : int 
            Numver of neaby "normal" columns to average over
    """
    ref = fits.open(ref_file)[0]
    data = ref.data * 1.
    mask = fits.open(mask_file)[0].data

    sind = np.where(np.nansum((mask & 4),axis=0)>0)[0]
    normals = np.where(np.nansum((mask & 4),axis=0)==0)[0]
    
    QE = np.ones_like(mask) * 1.
    corrected = deepcopy(data)
    for i in sind:
        closest = np.nanmin(abs(normals-i))
        nind = normals[abs(normals-i) < closest+av_size]
        s1 = fit_strap(data[:,i])
        n1 = fit_strap(np.nanmedian(data[:,nind],axis=1))
        factor = np.nanmedian(n1/s1)
        QE[:,i] = factor
        corrected[:,i] =  corrected[:,i] * factor
    
    cor_header = deepcopy(ref.header)
    cor_header['STRAPCAL'] = ('T', 'T if the strap QE has been rescaled')

    Make_fits(corrected, output, cor_header)
    name = output.split('.fits')[0] + '.qe.fits'
    Make_fits(QE, name, cor_header,integer=False) # scaled to make it work in integer form 

    return 'Rescaled {}, saved as {}'.format(ref_file,output)

from copy import deepcopy

def calc_strap_factor(i,breaks,size,normals,data):
    qe = np.ones_like(data) * 1. * np.nan
    b = breaks[i]
    nind = normals[b-av_size:b]
    eind = normals[b:b+av_size]
    nind = np.append(nind,eind) + 1
    norm = fit_strap(np.nanmedian(data[:,nind],axis=1))
    for j in range(size[i]):  
        s1 = fit_strap(data[:,normals[b]+1+j])
        factor = np.nanmedian(norm/s1)
        qe[:,normals[b]+1+j] = factor
    return qe

def correct_straps(ref_file,mask_file,ext,output,size=5,parallel=True):
    ref = fits.open(ref_file)[ext]
    data = ref.data * 1.
    mask = fits.open(mask_file)[0].data

    sind = np.where(np.nansum((mask & 4),axis=0)>0)[0]
    normals = np.where(np.nansum((mask & 4),axis=0)==0)[0]
    
    breaks = np.where(np.diff(normals,append=0)>1)[0]
    size = (np.diff(normals,append=0))[np.diff(normals,append=0)>1]

    if parallel:
        num_cores = multiprocessing.cpu_count()
        x = np.arange(0,len(breaks))
        qe = np.array(Parallel(n_jobs=num_cores)(delayed(calc_strap_factor)(i,breaks,size,normals,data) for i in x))
        qe = np.nanmedian(qe,axis=0)
        qe[np.isnan(qe)] = 1   
    else:
        qe = []
        for i in range(len(breaks)):
            qe += [calc_strap_factor(i,breaks,size,normals,data)]
            qe = np.array(qe)
            qe = np.nanmedian(qe,axis=0)
        qe[np.isnan(qe)] = 1   

    cor_header = deepcopy(ref.header)
    cor_header['STRAPCAL'] = ('T', 'T if the strap QE has been rescaled')

    Make_fits(data * qe, output, cor_header)
    name = output.split('.fits')[0] + '.qe.fits'
    Make_fits(qe, name, cor_header,integer=False) # scaled to make it work in integer form 

    return 'Rescaled {}, saved as {}'.format(ref_file,output)



def define_options(parser=None, usage=None, conflict_handler='resolve'):
    if parser is None:
        parser = argparse.ArgumentParser(usage=usage, conflict_handler=conflict_handler)

    parser.add_argument('-f','--ref_file', default = None, 
            help=('Reference fits file to rescale strap QE.'))
    parser.add_argument('-m','--mask', default = None,
            help=('Full pipeline mask for the image.'))
    parser.add_argument('-ext','--extension', default = 0,
            help=('Full pipeline mask for the image.'))
    parser.add_argument('-o','--output', default = 'default.rescale.fits',
            help=('Full output path/name for the rescaled image'))
    parser.add_argument('--av_size',default = 5,
            help=('number of nearby columns to average.'))
    parser.add_argument('--parallel',default = True,
            help=('use parallel processing.'))


    return parser

    
if __name__ == '__main__':
    print('Rescaling strap QE for TESS image')
    parser  = define_options()
    args    = parser.parse_args()
    print('got options: ',args)
    file    = args.ref_file
    mask    = args.mask
    ext     = int(args.extension)
    save    = args.output
    av_size = int(args.av_size)
    par     = args.parallel

    correct_straps(file, mask,ext, save, av_size,par)
    print('Rescaled {}, saved as {}'.format(file,save))
