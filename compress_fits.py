#!/usr/bin/env python

from astropy.io import fits
from glob import glob 
import argparse
import os

def Compress_fits(pattern, save_dir=None):
	files = glob(pattern)
	for f in files:
		if '.fz' not in f:
			try:
				hdu = fits.open(f)
			except:
				raise(ValueError('Could not open {} as fits file'.format(f)))
			if save_dir is None:
				name = f + '.fz'
			else:
				name = save_dir + f + '.fz'
				makedir(save_dir)
			hdu.writeto(name,overwrite=True)
			message = 'compressed {} to {}'.format(f, name)
			print(message)

def makedir(dir):
	try:
		if not os.path.exists(dir):
			os.makedirs(dir)
			print('Made dir ' + dir)
	except FileExistsError:
		pass
		

def define_options(parser=None, usage=None, conflict_handler='resolve'):
	if parser is None:
		parser = argparse.ArgumentParser(usage=usage, conflict_handler=conflict_handler)

	parser.add_argument('-p','--pattern', default = None, 
			help=('Files to compress, if using wildcards make sure its in quotes.'))
	parser.add_argument('-s','--save_dir', default = None,
			help=('directory to save compressed files to.'))
	return parser


if __name__ == '__main__':
	print('compressing')
	parser = define_options()
	args = parser.parse_args()
	print('got options: ',args)
	Compress_fits(args.pattern,args.save_dir)
	print('Compressed all as fits.fz')