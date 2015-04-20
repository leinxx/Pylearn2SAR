#!/usr/bin/env python

import sys
import os
import sys
import numpy as np
import gdal
from gdalconst import *
import scipy.signal as signal

def mirror_image(im, mask):
    def mirror_ud(im, mask):
        im = im * (1 - mask)
        mask = (mask != 0).astype(np.int8)
        diff = mask[0:-1,:] - mask[1:,:]
        rows = im.shape[0]
        cols = im.shape[1]
        uplist = np.where(diff == 1)
        for i, j in zip(uplist[0], uplist[1]):
             i = i + 1
             pad = im[i+1:i+min(rows - i - 1, i)+1, j]
             pad_mask = 1 - mask[i+1:i+min(rows - i - 1, i) + 1, j]
             pad_mask = np.flipud(pad_mask)
             pad = np.flipud(pad)
             im[i-pad.shape[0]:i,j] += pad * pad_mask * mask[i-pad.shape[0]:i,j];
             mask[i-pad.shape[0]:i,j] = mask[i-pad.shape[0]:i,j]*(1-pad_mask);

        return im, mask
    im2, mask2 = mirror_ud(im,mask)
    Image.fromarray(im2).show()
    im3, mask3 = mirror_ud(np.flipud(im),np.flipud(mask))
    im3 = np.flipud(im3)
    Image.fromarray(im3).show()
    mask3 = np.flipud(mask3)
    return im2 * (1 - mask2) + im3 * ((mask3 == 0) & (mask2 == 1))

def gen_land_free_data(image_dir, image_subfix, mask_subfix, output_dir):
    imalist = os.listdir(image_dir+'ima/')
    mask_dir = image_dir+'/mask/'
    for fname in imalist:
        if not fname.endswith('_ima.txt'):
            imalist.remove(fname)

    for fname in imalist:
        print fname
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        day = fname.split('_ima')[0]

        ds = gdal.Open(mask_dir + str(day) + mask_subfix, GA_ReadOnly)
        print mask_dir + str(day) + mask_subfix
        mask = ds.ReadAsArray()
        mask = (mask != 0)
        # expand land, to make sure land pixels are removed
        mask = signal.convolve2d(mask, np.ones((7, 7)), mode='same',\
            boundary='fill',fillvalue=0)
        mask = (mask > 0.001).astype(np.int8)
        for subfix in image_subfix:
            filein = image_dir + 'hhv/' + str(day) + subfix
            dataset = gdal.Open(filein, GA_ReadOnly )
            if dataset is None:
                print 'file does not exist:'
                print fname
                return
            image = dataset.ReadAsArray()
            image = mirror_image(image, mask)

            format = "GTiff"
            driver = gdal.GetDriverByName( format )
            dst_ds = driver.CreateCopy(output_dir+'/hhv_land_free/'+\
                str(day)+subfix, dataset, 0 )
            dst_ds.GetRasterBand(1).WriteArray(image)

            dst_ds = driver.CreateCopy(output_dir+'/mask_land_free/'+\
                str(day)+'-mask.tif', dataset, 0 )
            dst_ds.GetRasterBand(1).WriteArray(mask*255)

            dst_ds = None
            dataset = None
        ds = None

if __name__ == '__main__':
    image_dir = '/home/lein/Work/Sea_ice/gsl2014_hhv_ima/'
    image_subfix = ['-HH-8by8-mat.tif', '-HV-8by8-mat.tif']
    mask_subfix = '-mask.tif'
    gen_land_free_data(image_dir, image_subfix, mask_subfix, image_dir)
