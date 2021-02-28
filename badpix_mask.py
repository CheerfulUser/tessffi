#!/usr/bin/env python

import numpy as np
from astropy.io import fits
import argparse

def Make_fits(data, name, header):
    print('makefits shape ',data.shape)
    newhdu = fits.PrimaryHDU(data, header = header)
    newhdu.scale('int16', bscale=1.0,bzero=32768.0)
    newhdu.writeto(name,overwrite=True)
    return 

def Make_bad_pixel_mask(file,image,name):
    
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
        
    Make_fits(mask,name,header)


def define_options(parser=None, usage=None, conflict_handler='resolve'):
    if parser is None:
        parser = argparse.ArgumentParser(usage=usage, conflict_handler=conflict_handler)

    parser.add_argument('-f','--file', default = None, 
            help=('Fits file to make the mask of.'))
    parser.add_argument('-o','--output', default = 'default.badpix.fits',
            help=('Full output path/name for the created mask'))
    parser.add_argument('-i','--image',default = None,
            help=('example image, used to get dimensions.'))

    return parser



if __name__ == '__main__':
    print('Making bad pixel mask for TESS image')
    parser = define_options()
    args   = parser.parse_args()
    print('got options: ',args)
    file   = args.file
    save   = args.output
    image  = args.image
    if image is None:
        raise ValueError('Need an image to use for the header')

    Make_bad_pixel_mask(file, image, save)
    print('Made bad pixel mask for {}, saved as {}'.format(file,save))