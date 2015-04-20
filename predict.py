#!/usr/bin/env python
'''
  autors: Lei Wang, University of Waterloo
  email: alphaleiw@gmail.com
'''

import theano
import theano.tensor as T
from pylearn2.models import mlp
import time
from pylearn2.utils import serial
import numpy as np
import cPickle
import gdal
import math
import operator
import Image
import utils
import os
import gdal
from gdalconst import *
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from preprocessing_standarize import Standardize
import re

def lrange(num1, num2=None, step=1):
    op = operator.__lt__
    if num2 is None:
        num1, num2 = 0, num1
    if num2 < num1:
        if step > 0:
            num1 = num2
        op = operator.__gt__
    elif step < 0:
        num1 = num2
    while op(num1, num2):
        yield num1
        num1 += step

def sar_predict(ann,image,outfile):
    '''Predict on a image use a trianed model ann'''
    #define prediction function
    batch_size = ann.batch_size
    X = ann.get_input_space().make_batch_theano()
    y = ann.fprop(X)
    f = theano.function([X],y)
    #moving window
    window = ann.get_input_space().shape[0]
    rl = int(math.floor(window/2))
    rr = int(window-rl)
    #extract patch
    stride = 1
    jcol = range(rl, image.shape[2]-rr,stride)
    irow = range(rl, image.shape[1]-rr,stride)
    nrow = len(irow)
    ncol = len(jcol)
    batch_size = ann.batch_size
    yaml_src = ann.dataset_yaml_src.split()
    mean_std_file = yaml_src[yaml_src.index('mean_std_file:')+1]
    mean_std_file = re.search('\w+...\w+', mean_std_file).group(0)

    preprocessor = Standardize(mean_std_file = mean_std_file)
    # ann.set_batch_size(batch_size)
    m = len(jcol)*len(irow)
    extra = batch_size-m%batch_size

    im_pred_tmp = np.zeros(shape=(m,ann.get_output_space().dim))
    xshape = ann.get_input_space().shape
    subimgs = np.zeros(shape=(batch_size,ann.get_input_space().num_channels, xshape[0],xshape[1]),dtype=ann.get_input_space().dtype)
    for i in range(0,m-m%batch_size,batch_size):
        for j in range(i,i+batch_size):
            pos = [math.floor(j/ncol)*stride+irow[0],(j%ncol)*stride+jcol[0]]
            subimgs[j-i,:] = image[:,pos[0]-rl:pos[0]+rr,pos[1]-rl:pos[1]+rr]
        #the default input_space for pylearn convnet is (b,0,1,c)
        #the axis of subimgs should be permuted
        batch = DenseDesignMatrix(X = np.transpose(subimgs,(0,2,3,1)))
        preprocessor.apply(batch)
        im_pred_tmp[i:i+batch_size,:] = f(batch.X)
    im_pred_tmp = np.transpose(im_pred_tmp);
    im_pred_tmp = im_pred_tmp.reshape(im_pred_tmp.shape[0],nrow,ncol)

    #output tif
    def WriteArrayToTiff(array, outfile):
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(outfile,array.shape[2],array.shape[1],array.shape[0],
            GDT_Float32)
        nchannels = im_pred_tmp.shape[0]
        for band,i in zip(array,range(nchannels)):
            ds.GetRasterBand(i+1).WriteArray(array[i,:,:])
        ds.FlushCache()
        ds = None

    im_pred = np.zeros(shape=(im_pred_tmp.shape[0],image.shape[1],image.shape[2]))
    im_pred[:,rl:image.shape[1]-rr,rl:image.shape[2]-rr]=im_pred_tmp
    np.save(outfile+".pyn",im_pred)
    WriteArrayToTiff(im_pred,outfile)
    print 'prediction max is {}'.format(im_pred.max())


def imread(imagefile):

    dataset = gdal.Open(imagefile, GA_ReadOnly )
    if dataset is None:
        print 'File does not exist:'
        print imagefile
        return None
    im = dataset.ReadAsArray()
    im = im.astype(np.float32)/255.0
    return im


def read_hhv(input_path):
    '''
    Read hh and hv to a [len(input_path),nrow,ncol] numpy matrix

    '''
    im = imread(input_path[0])
    input_path = input_path[1:]
    if input_path is None:
        return im
    shape = im.shape
    for strfile in input_path:
        im = np.concatenate((im,imread(strfile)),axis=0)
    im=im.reshape(len(input_path)+1,shape[0],shape[1])
    return im

def predict_from_file(ann,input_path,out_path):
    im = read_hhv(input_path)
    if type(ann) == str:
        with open(ann) as f:
            ann = cPickle.load(f)
    assert ann.input_space.num_channels == im.shape[0]
    return sar_predict(ann,im,out_path)


def predict_from_date(ann,date,pred_path):
    global __predictDir__, __modelDir__, __datasetDir__, __sarImageDir__
    if isinstance(date,int):
        date = str(date)
    if type(ann) == str:
        with open(ann) as f:
            ann = cPickle.load(f)

    num_channels = ann.input_space.num_channels
    num_rows,num_cols = ann.input_space.shape
    subfix = ['-HH-8by8-mat.tif','-HV-8by8-mat.tif']
    input_path = (__sarImageDir__+date+subfix[0],__sarImageDir__+date+subfix[1])
    #pred_path = date+']_pred_conv_{0}_{1}_{2}_'.format(num_rows,num_cols,num_channels)+sufix+'.tif'
    return predict_from_file(ann,input_path,pred_path)


def test():
    import matplotlib.pyplot as plt
    kwargs = get_default_configure()
    for p in [kwargs['train_path'],kwargs['test_path'],kwargs['valid_path']]:
        ds = load_data_balance_under_sample(p,kwargs['num_rows'],kwargs['num_cols'],kwargs['num_channels'])
        trn = SarDataset(np.array(ds[0]),ds[1])
        plt.figure()
        plt.imshow(trn.X[100,0,:,:])
        figure
        plt.hist(trn.y)


if __name__ == '__main__':
    import sys
    if len(sys.argv != 4):
      print '''Usage: ./predict ann_file input_images output_dir
        ann_file:
        input_images: a txt file with each input scene as one line, 
            when multiple bands (such as HH and HV) are in different files, 
            list them in the same line. A example input image file:
            im1_b1.tif im1_b2.tif
            im2_b1.tif im2_b2.tif
        output_dir: the output directory
      '''
    else:
      ann_file = sys.argv[1]
      input_images = sys.argv[2]
      output_dir = sys.argv[3]
      ann = cPickle.load(ann_file)
      infiles = open(input_images,'r').readlines()
      infiles = [line.strip().split() for line in inlines]
      for f in infiles:
          output_file = output_dir + '/' + f[0].split('.')[0:-1] + '.predict.tif'
          predict_image(ann, f, output_file)
