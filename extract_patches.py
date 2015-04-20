#!/usr/bin/env python

"""
This script prepare data to be used in the convolutional neural network for the SAR image sea ice concentration estimation problem.

Each SAR image is separated to small patches and seriealized to a plz file. Three objects should be dumped:
#train_set, valid_set, test_set format: tuple(input, target)
#input is an np.ndarray of 2 dimensions (a matrix)
#witch row's correspond to an example. target is a
#np.ndarray of 1 dimensions (vector)) that have the same length as
#the number of rows in the input. It should give the target
#target to the example with the same index in the input.

The image analysis is to be used for training and verification
"""
import sys
import cPickle
import gzip
import os
import sys
import Image
import numpy as np
import math
import gdal
from gdalconst import *


def export_pkl(inputs,target, outname):
    index = np.arange(target.size)
    default_seed = 0
    np.random.seed(0)
    np.random.shuffle(index)
    inputs = np.transpose(inputs[index],axes = [0,2,3,1])
    target = target[index]
    train_set = (inputs, target)
    with open(outname,'wb') as fp:
        cPickle.dump(train_set, fp,0)



def patch_img(image, points, window):
    #img is a image matrix
    #samples are the locations to extract patches
    #window is the patchsize, should be a odd number, if not, it will be transfered to a odd number by +1

    # check windows size
    if window%2 == 1:
        window = window+1
        print 'adjust window size to {0}'.format(window)
    rl = math.floor(window/2)-1
    rr = window-1-rl

    #select available points:
    #conditions: not masked as 0, in image, not out of boundary

    inputs = []
    target = []
    subpoints = []
    for i,point in zip(range(len(points)), points):
         if point[0] >= rl and point[0] < image.shape[2]-rr and \
            point[1] >= rl and point[1] < image.shape[1]-rr :
            inputs.append(image[:,int(point[1])-rl:int(point[1])+rr+1,
            int(point[0])-rl:int(point[0])+rr+1])
            target.append( point[2])
            subpoints.append(point)
    return np.asarray(inputs), np.asarray(target),np.asarray(subpoints)

def run(window, basefolder, image_subfixes, output_dir):
    imalist = os.listdir(basefolder+'ima/')
    for fname in imalist:
        if not fname.endswith('_ima.txt'):
            imalist.remove(fname)

    for fname in imalist:
        print fname
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        day = fname.split('_ima')[0]
        inputs = []
        target = []
        outname = output_dir+str(day)+'-HHV-8by8-patch.pkl'
        hhv = []
        for subfix in image_subfixes:
            filein = basefolder+ 'hhv_land_free/' + str(day) + subfix
            dataset = gdal.Open(filein, GA_ReadOnly )
            if dataset is None:
                print 'file does not exist:'
                print fname
                return
            image = dataset.ReadAsArray()
            image = image.astype(np.float32)/255.0
            hhv.append(image)
        hhv = np.asarray(hhv)
        ima = []
        with open(basefolder+'ima/'+fname) as f:
            for line in f:
                point = map(float,line.split(' '))
                ima.append(point)
        inputs, target, subpoints = patch_img(hhv, ima, window)
        assert(len(inputs)==len(target)==len(subpoints))
        if not os.path.exists(output_dir+'ima_used'):
            os.makedirs(output_dir+'ima_used')
        np.savetxt(output_dir+'ima_used/'+str(day)+'_ima_used.txt',subpoints,fmt='%.2f')
        export_pkl(inputs,target, outname)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        window=int(sys.argv[1])
    else:
        window = 40
    print 'window size is set to {0}'.format(window)

    basefolder = '/home/lein/Work/Sea_ice/gsl2014_hhv_ima/'
    image_subfixes = ['-HH-8by8-mat.tif', '-HV-8by8-mat.tif']
    output_dir = '../dataset/gsl2014_land_free_{}/'.format(window)
    #basefolder = '/home/lein/Work/sar_dnn/dataset/beaufort_2010_2011/'
    #image_subfixes = ['-HH-8by8-mat.tif', '-HV-8by8-mat.tif']
    #output_dir = '/home/lein/Work/sar_dnn/dataset/beaufort_2010_2011/batches_{}'.format(window)
    run(window, basefolder, image_subfixes, output_dir)
