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


def region_cut(table,wcs):
    ra = table.ra.values
    dec = table.dec.values
    foot = wcs.calc_footprint()
    minra = min(foot[:,0])
    maxra = max(foot[:,0])
    mindec = min(foot[:,1])
    maxdec = max(foot[:,1])
    inddec = (dec < maxdec) & (dec> mindec)
    indra = (ra < maxra) & (ra> minra)
    ind = indra * inddec
    tab = table.iloc[ind]
    return tab

def circle_app(rad):
    """
    makes a kinda circular aperture, probably not worth using.
    """
    mask = np.zeros((int(rad*2+.5)+1,int(rad*2+.5)+1))
    c = rad
    x,y =np.where(mask==0)
    dist = np.sqrt((x-c)**2 + (y-c)**2)

    ind = (dist) < rad + .2
    mask[y[ind],x[ind]]= 1
    return mask

def check_table_format(table): 
    try:
        temp = table.x
        temp = table.y
        temp = table.ra
        temp = table.dec
        temp = table.radius
        temp = table.mag
        temp = table.mjd_start
        temp = table.mjd_end
    except:
        message = ("mask_table must be a csv with the following columns:\nx\ny\nra\ndec\nradius\nmag\nmjd_start\nmjd_end\n"
                    + "Only a position (x,y) or (ra,dec) and size (radius) or (mag) is needed to run.")
        raise ValueError()

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


def region_cut(table,wcs):
    ra = table.ra
    dec = table.dec
    foot = wcs.calc_footprint()
    minra = min(foot[:,0])
    maxra = max(foot[:,0])
    mindec = min(foot[:,1])
    maxdec = max(foot[:,1])
    inddec = (dec < maxdec) & (dec> mindec)
    indra = (ra < maxra) & (ra> minra)
    ind = indra * inddec
    tab = table.iloc[ind]
    return tab

def circle_app(rad):
    """
    makes a kinda circular aperture, probably not worth using.
    """
    mask = np.zeros((int(rad*2+.5)+1,int(rad*2+.5)+1))
    c = rad
    x,y =np.where(mask==0)
    dist = np.sqrt((x-c)**2 + (y-c)**2)

    ind = (dist) < rad + .2
    mask[y[ind],x[ind]]= 1
    return mask

def check_table_format(table): 
    try:
        temp = table.x
        temp = table.y
        temp = table.ra
        temp = table.dec
        temp = table.radius
        temp = table.mag
        temp = table.mjd_start
        temp = table.mjd_end
    except:
        message = ("mask_table must be a csv with the following columns:\nx\ny\nra\ndec\nradius\nmag\nmjd_start\nmjd_end\n"
                    + "Only a position (x,y) or (ra,dec) and size (radius) or (mag) is needed to run.")
        raise ValueError()

def Spot_mask(fits_file,mask_table,ext=0):
    table = pd.read_csv(mask_table)
    check_table_format(table)
    
    hdu = fits.open(fits_file)[ext]
    # uses the file name to set the time, not versitile 
    t = float(fits_file.split('/')[-1].split('_')[1])
    image = hdu.data
    wcs = WCS(hdu.header)
    spotmask = np.zeros_like(image,dtype=float)
    for i in range(len(table)):
        row = table.iloc[i]
        start = row.mjd_start
        end   = row.mjd_end
        cont = True

        if np.isfinite(start):
            if t < start:
                cont = False
        if np.isfinite(end):
            if t > end:
                cont = False
        if cont:
            if np.isfinite(row.x) & np.isfinite(row.y):
                x = int(row.x + 0.5)
                y = int(row.y + 0.5)
            elif np.isfinite(row.ra) & np.isfinite(row.dec):
                x,y = wcs.all_world2pix(row.ra,row.dec,0)
                if size_limit(x,y,image):
                    pass
                else:
                    x = np.nan
                    y = np.nan
                    print('coordinates ra={}, dec={} not in range'.format(np.round(ra,2),np.round(row.dec,2)))
                    pass
            else:
                print('no position provided')
            # make aperture time
            rad = row.radius
            mag = row.mag
            if np.isfinite(rad):
                ap = circle_app(rad)

                temp = np.zeros_like(image)
                temp[y,x] = 1
                conv = fftconvolve(temp, ap,mode='same')#.astype(int)
                temp = (conv > 0.9) * 1.
                spotmask += conv 

            elif np.isfinite(mag):
                mags = np.array([18,17,16,15,14,13.5,12,10,9,8,7])
                size = (np.array([3,4,5,6,7,8,10,14,16,18])).astype(int)
                diff = mag - mags
                ind = np.where(diff < 0)[0][-1]
                ap = circle_app(size[ind])

                temp = np.zeros_like(image)
                temp[y,x] = 1
                conv = fftconvolve(temp, ap,mode='same')#.astype(int)
                temp = (conv > 0.5) * 1
                spotmask += conv 

            else:
                print('no radius or magnitude provided')
    spotmask = (spotmask >= .5).astype(int) * 64

    return spotmask