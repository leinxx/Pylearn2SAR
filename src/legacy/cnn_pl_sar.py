#!/usr/bin/env python
import theano
import theano.tensor as T
from pylearn2.models import mlp
from pylearn2.training_algorithms import sgd
from pylearn2.termination_criteria import EpochCounter
from pylearn2 import termination_criteria
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.train_extensions import best_params
from pylearn2.train_extensions import window_flip
from pylearn2.train import Train
from pylearn2 import space
from pylearn2 import monitor
from pylearn2.costs.mlp import dropout as dropout
from pylearn2.costs.mlp import WeightDecay
import pylearn2.models.maxout as maxout
import pylearn2.costs.cost as cost
import time
from pylearn2.utils import serial
import numpy as np
import sarTransformerDataset
import cPickle
import gdal
import math
import operator
import Image
import utils
import os
import pprint
import gdal
from gdalconst import *
__sarImageDir__ = '/home/lein/Work/Sea_ice/after-reproj/'
__modelDir__ = '../models/'
__predictDir__ = '../output/'
__datasetDir__ = '../dataset/sar_hhv_pkl_41/'


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


def gen_center_sub_window(window, sub_size):
    '''
    Generate the center window of size sub_size in a window of size windw.
    '''
    c = math.ceil(window/2)-1
    rl = math.ceil(sub_size/2)-1
    return [c-rl, c-rl, sub_size, sub_size]

def load_data_balance_under_sample(datapath, num_rows, num_cols, num_channels):
    if isinstance(datapath,str):
        datapath = (datapath,)
    datasets = []
    for p in datapath:
        with open(p) as f:
            if datasets == []:
                datasets = cPickle.load(f)
                train_x, train_y = datasets[0]
                valid_x,valid_y = datasets[1]
                test_x, test_y = datasets[2]
                train_y = train_y.reshape(len(train_y),1)#.astype('float64')
                valid_y = valid_y.reshape(len(valid_y),1)#.astype('float64')
                test_y = test_y.reshape(len(test_y),1)#.astype('float64')
            else:
                ds = cPickle.load(f)
                train_x = np.concatenate((train_x,ds[0][0]),axis=0)
                train_y = np.concatenate((train_y,ds[0][1].reshape(len(ds[0][1]),1)),axis=0)
                valid_x = np.concatenate((valid_x,ds[1][0]),axis=0)
                valid_y = np.concatenate((valid_y,ds[1][1].reshape(len(ds[1][1]),1)),axis=0)
                test_x = np.concatenate((test_x,ds[2][0]),axis=0)
                test_y = np.concatenate((test_y,ds[2][1].reshape(len(ds[2][1]),1)),axis=0)
    train_x = train_x.reshape(len(train_x),num_channels,num_rows,num_cols)
    valid_x = valid_x.reshape(len(valid_x),num_channels,num_rows,num_cols)
    test_x =  test_x.reshape(len(test_x), num_channels,num_rows,num_cols)
    X,y = np.concatenate((train_x,test_x,valid_x),axis=0), np.concatenate((train_y,test_y,valid_y),axis=0)

    #balance the training samples to make the number of samples for each ice concentration value is roughly the same
    hist = np.histogram(y,bins=np.arange(-0.05,1.06,0.1))
    maxi = hist[0].max() # this is water
    hist = hist[0][1:]
    mean_ice = hist[hist>0].mean() # this is the largest amount of ice samples at a ice concentration level
    nfold = int(maxi/mean_ice)
    fold_size = maxi/nfold
    # shuffle the data and select batches
    iwater = np.where(y<0.05)[0]
    iice = np.where(y>0.05)[0]
    #np.random.RandomState(seed=0).shuffle(index)
    which_fold = load_data_balance_under_sample.which_fold
    iwater = iwater[(which_fold)*fold_size:min(((which_fold)+1)*fold_size,len(iwater))]
    index = np.concatenate((iwater,iice))
    index.sort()
    xx = X[index]
    yy = y[index]
    load_data_balance_under_sample.which_fold += 1
    load_data_balance_under_sample.which_fold = load_data_balance_under_sample.which_fold%nfold
    print nfold
    return xx,yy
load_data_balance_under_sample.which_fold = 0

def under_sample_water(X,y):
    hist = np.histogram(y,bins=np.arange(-0.05,1.06,0.1))
    maxi = hist[0].max() # this is water
    hist = hist[0][1:]
    mean_ice = hist[hist>0].mean() # this is the largest amount of ice samples at a ice concentration level
    nfold = int(maxi/mean_ice)
    fold_size = maxi/nfold
    # shuffle the data and select batches
    iwater = np.where(y<0.05)[0]
    iice = np.where(y>0.05)[0]
    #np.random.RandomState(seed=0).shuffle(index)
    which_fold = under_sample_water.which_fold
    iwater = iwater[(which_fold)*fold_size:min(((which_fold)+1)*fold_size,len(iwater))]
    index = np.concatenate((iwater,iice))
    index.sort()
    xx = X[index]
    yy = y[index]
    under_sample_water.which_fold += 1
    under_sample_water.which_fold = under_sample_water.which_fold%nfold
    print nfold
    return xx,yy
under_sample_water.which_fold = 0

def load_data_transformed(datapath,num_rows,batch_size,under_sample=True):
    ds = sarTransformerDataset.TransformerDataset(num_rows,batch_size,datapath=datapath)
    data = ds.get_samples_all()
    if under_sample is True:
        data = under_sample_water(*data)
    return data


def load_data_balance(datapath,num_rows,num_cols,num_channels):

    if isinstance(datapath,str):
        datapath = (datapath,)
    datasets = []
    for p in datapath:
        with open(p) as f:
            if datasets == []:
                datasets = cPickle.load(f)
                train_x, train_y = datasets[0]
                valid_x,valid_y = datasets[1]
                test_x, test_y = datasets[2]
                train_y = train_y.reshape(len(train_y),1)#.astype('float64')
                valid_y = valid_y.reshape(len(valid_y),1)#.astype('float64')
                test_y = test_y.reshape(len(test_y),1)#.astype('float64')
            else:
                ds = cPickle.load(f)
                train_x = np.concatenate((train_x,ds[0][0]),axis=0)
                train_y = np.concatenate((train_y,ds[0][1].reshape(len(ds[0][1]),1)),axis=0)
                valid_x = np.concatenate((valid_x,ds[1][0]),axis=0)
                valid_y = np.concatenate((valid_y,ds[1][1].reshape(len(ds[1][1]),1)),axis=0)
                test_x = np.concatenate((test_x,ds[2][0]),axis=0)
                test_y = np.concatenate((test_y,ds[2][1].reshape(len(ds[2][1]),1)),axis=0)
    train_x = train_x.reshape(len(train_x),num_channels,num_rows,num_cols)
    valid_x = valid_x.reshape(len(valid_x),num_channels,num_rows,num_cols)
    test_x =  test_x.reshape(len(test_x), num_channels,num_rows,num_cols)
    X,y = np.concatenate((train_x,test_x,valid_x),axis=0), np.concatenate((train_y,test_y,valid_y),axis=0)

    #balance the training samples to make the number of samples for each ice concentration value is roughly the same
    hist = np.histogram(y,bins=np.arange(-0.05,1.06,0.1))
    maxi = hist[0].argmax()
    index = np.logical_and(y>hist[1][maxi] , y < hist[1][maxi+1]).ravel()

    maxn = index.sum()
    xx = X[index,:,:,:]
    yy = y[index,:]
    for i in range(1,11):
        index = np.logical_and(y > hist[1][i] , y <hist[1][i+1]).ravel()
        r = int(maxn/index.sum())
        xx = np.concatenate((xx,np.repeat(X[index,:,:,:],r,axis=0)),axis=0)
        yy = np.concatenate((yy,np.repeat(y[index,:],r,axis=0)),axis=0)
    #test
    hist = np.histogram(y,bins=np.arange(-0.05,1.06,0.1))
    return xx,yy



def load_data(datapath,num_rows,num_cols,num_channels):

    if isinstance(datapath,str):
        datapath = (datapath,)
    train_x = []
    train_y = []
    for p in datapath:
        with open(p) as f:
            if train_x == []:
                train_x, train_y = cPickle.load(f)
                train_y = train_y.reshape(len(train_y),1)#.astype('float64')
            else:
                ds = cPickle.load(f)
                train_x = np.concatenate((train_x,ds[0]),axis=0)
                train_y = np.concatenate((train_y,ds[1].reshape(len(ds[1]),1)),axis=0)
    #train_x = train_x.reshape(len(train_x),num_channels,num_rows,num_cols)
    return train_x, train_y

class SarDataset(DenseDesignMatrix):
    def __init__(self,X,y,sub_window=None):
        '''
        X: ['b','c',0,1]
        sub_window: [top, left, height,width] the window that actually needed in the center
        '''
        if sub_window != None:
            X = X[:,:,sub_window[0]:sub_window[0]+sub_window[2],sub_window[1]:sub_window[1]+sub_window[3]]
        X = np.transpose(X,axes = [0,2,3,1])
        super(SarDataset,self).__init__(topo_view=X,y=y,axes = ('b', 0, 1,'c'))
        #super(SarDataset,self).__init__(topo_view=X,y=y,axes = ('b', 'c',0, 1))



class SarDatasetfromFiles(DenseDesignMatrix):
    def __int__(self, dataset_dir, num_rows,num_cols, num_channels,fns=None,which_set=None):

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_channels = num_channels

        if dataset_dir[-1] != '/':
            dataset_dir = dataset_dir+'/'

        sufix = '-HH-8by8-mat.tif'
        traindays = numpy.array([
                '20100730',
                '20100806',
                '20100822',
                '20100829',
                '20100909',
                '20100929',
                '20101003',
                '20101006',
                '20101008'])
        testdays = numpy.array([
                '20110709',
                '20110710',
                '20110720'])
        validdays = numpy.array([
                '20110725',
                '20110811',
                '20110817'])
        trainset = [dataset_dir+day+sufix for day in traindays]
        testset = [dataset_dir+day+sufix for day in testdays]
        validset = [dataset_dir+day+sufix for day in validdays]
        if which_set == 'train':
            X,y = load_data(trainset, num_rows,num_cols, num_channels)
        elif which_set == 'valid':
            X,y = load_data(validset, num_rows,num_cols, num_channels)
        elif which_set == 'test':
            X,y = load_data(testset, num_rows,num_cols, num_channels)
        elif fns is not None:
            fns = [dataset_dir+day for day in fns]
            X,y = load_data(fns, num_rows,num_cols, num_channels)
        X = np.transpose(X,axes = [0,2,3,1])
        super(SarDatasetfromFiles,self).__init__(topo_view=X,y=y,axes = ('b', 0, 1,'c'))



class NonsymetricCost(cost.DefaultDataSpecsMixin, cost.Cost):
    def __init__(self):
        self.supervised = True



    def expr(self, model, data, **kwargs):
        self.get_data_specs(model)[0].validate(data)
        (X,Y)=data
        return T.abs_(Y-model(X))+10*T.maximum(0,model(X)-1)


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
    batch_size = 128
    ann.set_batch_size(batch_size)
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
        im_pred_tmp[i:i+batch_size,:] = f(np.transpose(subimgs,(0,2,3,1)))
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
    return im_pred



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

'''
    Verson 0 that doesn't work, but settings works
'''


def cnn_run_dropout_maxout(data_path,num_rows,num_cols,num_channels,input_path,pred_path):
    t = time.time();
    sub_window = gen_center_sub_window(76,num_cols)
    trn = SarDataset(ds[0][0],ds[0][1],sub_window)
    vld = SarDataset(ds[1][0],ds[1][1],sub_window)
    tst = SarDataset(ds[2][0],ds[2][1],sub_window)
    print 'Take {}s to read data'.format( time.time()-t)
    t = time.time()
    batch_size = 100
    h1 = maxout.Maxout(layer_name='h2', num_units=1,num_pieces=100,irange=.1)
    hidden_layer = mlp.ConvRectifiedLinear(layer_name='h2',output_channels=8,irange=0.05,kernel_shape=[5,5],pool_shape=[2,2],pool_stride=[2,2],max_kernel_norm=1.9365)
    hidden_layer2 = mlp.ConvRectifiedLinear(layer_name='h3',output_channels=8,irange=0.05,kernel_shape=[5,5],pool_shape=[2,2],pool_stride=[2,2],max_kernel_norm=1.9365)
    #output_layer = mlp.Softplus(dim=1,layer_name='output',irange=0.1)
    output_layer = mlp.Linear(dim=1,layer_name='output',irange=0.05)
    trainer = sgd.SGD(learning_rate=0.001,batch_size=100,termination_criterion=EpochCounter(2000),
                      cost = dropout.Dropout(),
                      train_iteration_mode='even_shuffled_sequential',
                      monitor_iteration_mode='even_shuffled_sequential',
                      monitoring_dataset={'test':  tst,
                                          'valid': vld,
                                          'train': trn})
    layers = [hidden_layer,hidden_layer2,output_layer]
    input_space = space.Conv2DSpace(shape=[num_rows,num_cols],num_channels=num_channels)

    ann = mlp.MLP(layers,input_space=input_space,batch_size=batch_size)
    watcher = best_params.MonitorBasedSaveBest(
            channel_name='valid_objective',
            save_path='sar_cnn_mlp.pkl')
    experiment = Train(dataset=trn,
                       model=ann,
                       algorithm=trainer,
                       extensions=[watcher])
    print 'Take {}s to compile code'.format(time.time()-t)
    t = time.time()
    experiment.main_loop()
    print 'Training time: {}s'.format(time.time()-t)
    serial.save('cnn_hhv_{0}_{1}.pkl'.format(num_rows,num_cols),ann,on_overwrite='backup')

    #read hh and hv into a 3D numpy
    image = read_hhv(input_path)
    return ann, sar_predict(ann,image,pred_path)


class linear_mlp_bayesian_cost(mlp.Linear):
    r = np.array(2.5).astype(np.float32)
    def cost(self,Y,Y_hat):
        w = T.fscalar()
        r = self.r
        w = 0.05
        i = T.le(Y,w)
        j = T.eq(i,0)
        z = T.join(0,Y[i]/r,Y[j])
        z_hat = T.join(0,Y_hat[i]/r,Y_hat[j])
        return super(linear_mlp_bayesian_cost,self).cost(z,z_hat)

class linear_mlp_ace(mlp.Linear):
    def cost(self,Y,Y_hat):
        return -(Y*T.log(Y_hat)+(1-Y)*T.log(1-Y_hat)).mean()

def cnn_train(train_path, test_path, valid_path, save_path, predict_path,image_path,num_rows=28,
             num_cols =28,
             num_channels =2,
             batch_size =128,
             output_channels =[64,64],
             kernel_shape =[[12,12],[5,5]],
             pool_shape =[[4,4],[2,2]],
             pool_stride =[[2,2],[2,2]],
             irange =[0.05,0.05,0.05],
             max_kernel_norm =[1.9365,1.9365],
             learning_rate =0.001,
             init_momentum =0.9,
             weight_decay =[0.0002,0.0002,0.0002],
             n_epoch = 1000,
             ):
    #load data
    #t = time.time()
    ds = load_data(valid_path, num_rows,num_cols, num_channels)
    vld = SarDataset(np.array(ds[0]),ds[1])
    ds = load_data(train_path, num_rows,num_cols, num_channels)
    trn = SarDataset(np.array(ds[0]),ds[1])
    ds = load_data(test_path, num_rows,num_cols, num_channels)
    tst = SarDataset(np.array(ds[0]),ds[1])
    #load balanced data
    #ds = load_data_balance_under_sample(train_path, num_rows,num_cols, num_channels)
    #trn = SarDataset(np.array(ds[0]),ds[1])
    #ds = load_data_balance(valid_path, num_rows,num_cols, num_channels)
    #vld = SarDataset(np.array(ds[0]),ds[1])
    #ds = load_data_balance(test_path, num_rows,num_cols, num_channels)
    #tst = SarDataset(np.array(ds[0]),ds[1])
    #print 'Take {}s to read data'.format( time.time()-t)
    #use gaussian convlution on the origional image to see if it can concentrate in the center
    #trn,tst,vld = load_data_lidar()

    #mytransformer = transformer.TransformationPipeline(input_space=space.Conv2DSpace(shape=[num_rows,num_cols],num_channels=num_channels),transformations=[transformer.Rotation(),transformer.Flipping()])
    #trn = contestTransformerDataset.TransformerDataset(trn,mytransformer,space_preserving=True)
    #tst = contestTransformerDataset.TransformerDataset(tst,mytransformer,space_preserving=True)
    #vld = contestTransformerDataset.TransformerDataset(vld,mytransformer,space_preserving=True)

    #trn = transformer_dataset.TransformerDataset(trn,mytransformer,space_preserving=True)
    #tst = transformer_dataset.TransformerDataset(tst,mytransformer,space_preserving=True)
    #vld = transformer_dataset.TransformerDataset(vld,mytransformer,space_preserving=True)

    #setup the network
    t = time.time()
    layers = []
    for i in range(len(output_channels)):
        layer_name = 'h{}'.format(i+1)
        convlayer = mlp.ConvRectifiedLinear(layer_name=layer_name, output_channels=output_channels[i],irange=irange[i],kernel_shape=kernel_shape[i],pool_shape=pool_shape[i],pool_stride=pool_stride[i],max_kernel_norm=max_kernel_norm[i])
        layers.append(convlayer)


    output_mlp = mlp.Linear(dim=1,layer_name='output',irange=irange[-1], use_abs_loss=True)
    #output_mlp = mlp.linear_mlp_ace(dim=1,layer_name='output',irange=irange[-1])
    layers.append(output_mlp)


    #ann = cPickle.load(open('../output/train_with_2010_2l_40_64/original_500/f/f0.pkl'))
    #layers = []
    #for layer in ann.layers:
    #    layer.set_mlp_force(None)
    #    layers.append(layer)

    trainer = sgd.SGD(learning_rate=learning_rate,batch_size=batch_size,
                      termination_criterion=EpochCounter(n_epoch),
                      #termination_criterion = termination_criteria.And([termination_criteria.MonitorBased(channel_name = 'train_objective', prop_decrease=0.01,N=10),EpochCounter(n_epoch)]),
                      #cost = dropout.Dropout(),
                      cost = cost.SumOfCosts([cost.MethodCost('cost_from_X'), WeightDecay(weight_decay)]),
                      init_momentum=init_momentum,
                      train_iteration_mode='even_shuffled_sequential',
                      monitor_iteration_mode='even_shuffled_sequential',
                      monitoring_dataset={'test':  tst,
                                          'valid': vld,
                                          'train': trn})

    input_space = space.Conv2DSpace(shape=[num_rows,num_cols],num_channels=num_channels)
    #ann = mlp.MLP(layers,input_space=input_space,batch_size=batch_size)
    ann = serial.load('../output/train_with_2010_2l_40_64/original_500/f/f0.pkl')
    ann = monitor.push_monitor(ann,'stage_0')
    watcher = best_params.MonitorBasedSaveBest(
            channel_name='valid_objective',
            save_path = predict_path+save_path)
    flip = window_flip.WindowAndFlip((num_rows,num_cols),randomize=[tst,vld,trn])
    experiment = Train(dataset=trn,
                       model=ann,
                       algorithm=trainer,
                       extensions=[watcher,flip])
    print 'Take {}s to compile code'.format(time.time()-t)

    #train the network
    t = time.time()
    experiment.main_loop()
    print 'Training time: {}h'.format((time.time()-t)/3600)
    utils.sms_notice('Training time:{}'.format((time.time()-t)/3600))

    return ann


def cnn_train_tranformer(train_path, test_path, valid_path, save_path, predict_path,num_rows=28,
             num_cols =28,
             num_channels =2,
             batch_size =128,
             output_channels =[64,64],
             kernel_shape =[[12,12],[5,5]],
             pool_shape =[[4,4],[2,2]],
             pool_stride =[[2,2],[2,2]],
             irange =[0.05,0.05,0.05],
             max_kernel_norm =[1.9365,1.9365],
             learning_rate =0.001,
             init_momentum =0.9,
             weight_decay =[0.0002,0.0002,0.0002],
             n_epoch = 1000,
             image_path = ''
        ):


    ds = load_data_transformed(train_path, num_cols,batch_size)
    ds = (np.transpose(ds[0],axes=[0,3,1,2]),ds[1])
    trn = SarDataset(np.array(ds[0]),ds[1])
    ds = load_data_transformed(valid_path, num_cols,batch_size)
    ds = (np.transpose(ds[0],axes=[0,3,1,2]),ds[1])
    vld = SarDataset(np.array(ds[0]),ds[1])
    ds = load_data_transformed(test_path, num_cols,batch_size)
    ds = (np.transpose(ds[0],axes=[0,3,1,2]),ds[1])
    tst = SarDataset(np.array(ds[0]),ds[1])
    #setup the network
    #X = np.random.random([400000,2,41,41])
    #y = np.random.random([400000,1])
    #trn = SarDataset(X,y)
    #X = np.random.random([60000,2,41,41])
    #y = np.random.random([60000,1])
    #tst = SarDataset(X,y)
    #X = np.random.random([60000,2,41,41])
    #y = np.random.random([60000,1])
    #vld = SarDataset(X,y)
    t = time.time()
    layers = []
    for i in range(len(output_channels)):
        layer_name = 'h{}'.format(i+1)
        convlayer = mlp.ConvRectifiedLinear(layer_name=layer_name, output_channels=output_channels[i],irange=irange[i],kernel_shape=kernel_shape[i],pool_shape=pool_shape[i],pool_stride=pool_stride[i],max_kernel_norm=max_kernel_norm[i])
        layers.append(convlayer)


    output_mlp = mlp.Linear(dim=1,layer_name='output',irange=irange[-1])
    #output_mlp = mlp.linear_mlp_bayesian_cost(dim=1,layer_name='output',irange=irange[-1])
    layers.append(output_mlp)

    trainer = sgd.SGD(learning_rate=learning_rate,batch_size=batch_size,
                      termination_criterion=EpochCounter(n_epoch),
                      #termination_criterion = termination_criteria.And([termination_criteria.MonitorBased(channel_name = 'train_objective', prop_decrease=0.01,N=10),EpochCounter(n_epoch)]),
                      #cost = dropout.Dropout(),
                      cost = cost.SumOfCosts([cost.MethodCost('cost_from_X'), WeightDecay(weight_decay)]),
                      init_momentum=init_momentum,
                      train_iteration_mode='even_shuffled_sequential',
                      monitor_iteration_mode='even_shuffled_sequential',
                      monitoring_dataset={'test':  tst,
                                          'valid': vld,
                                          'train': trn})

    input_space = space.Conv2DSpace(shape=[num_rows,num_cols],num_channels=num_channels)
    #ann = mlp.MLP(layers,input_space=input_space,batch_size=batch_size)
    watcher = best_params.MonitorBasedSaveBest(
            channel_name='valid_objective',
            save_path = predict_path+save_path)
    #flip = window_flip.WindowAndFlip((num_rows,num_cols),randomize=[tst,vld,trn])
    experiment = Train(dataset=trn,
                       model=ann,
                       algorithm=trainer,
                       extensions=[watcher])
    print 'Take {}s to compile code'.format(time.time()-t)

    #train the network
    t = time.time()
    experiment.main_loop()
    print 'Training time: {}h'.format((time.time()-t)/3600)
    utils.sms_notice('Training time:{}'.format((time.time()-t)/3600))

    return ann



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


def get_default_configure_gsl2014(window=40,nlayer=2,nfilter=64):

    import os
    global __predictDir__, __modelDir__, __datasetDir__, __sarImageDir__
    __sarImageDir__ = '/home/lein/Work/Sea_ice/gsl2014_hhv_ima/hhv/'
    dataset_dir = '../dataset/gsl2014_{}'.format(str(window))+'/'
    # 2 generate dataset if not existing
    #if not os.path.exists(__datasetDir__):
    #    extract_patches.prepare_data_window(window)

    predict_path = __predictDir__+'gsl2014_{}/'.format(window)

    if not os.path.exists(predict_path):
        os.makedirs(predict_path)

    save_path = '0.pkl'

    flist = os.listdir(dataset_dir)
    days = [f.split('-')[0] for f in flist if f.endswith('.pkl')]
    subfix = ['-HH-8by8-mat.tif','-HV-8by8-mat.tif']
    image_path = [(__sarImageDir__+day+'-HH-8by8-mat.tif',
    __sarImageDir__+day+'-HV-8by8-mat.tif') for day in days]

    train_path = [dataset_dir+day+'-HHV-8by8-patch.pkl' for day in days[0:18]]
    test_path = [dataset_dir+day+'-HHV-8by8-patch.pkl' for day in days[18:22]]
    valid_path = [dataset_dir+day+'-HHV-8by8-patch.pkl' for day in days[22:26]]
    with open('train.txt','w') as f:
        f.write('\n'.join(train_path))
    with open('test.txt','w') as f:
        f.write('\n'.join(test_path))
    with open('valid.txt','w') as f:
        f.write('\n'.join(valid_path))


    if nlayer == 2:
        return {
        'num_rows':window,
        'num_cols':window,
        'num_channels':2,
        'batch_size':128,
        'output_channels':[nfilter,nfilter*2],
        'kernel_shape':[[5,5],[5,5]],
        'pool_shape':[[2,2],[2,2]],
        'pool_stride':[[2,2],[2,2]],
        'irange':[0.05,0.05,0.05],
        'max_kernel_norm':[1.9365,1.9365],
        'learning_rate':0.003,
        'init_momentum':0.9,
        'weight_decay':[0.0005,0.0005,0.0005],
        'n_epoch': 200,
        'train_path':train_path,
        'test_path':test_path,
        'valid_path':valid_path,
        'save_path':save_path,
        'predict_path':predict_path,
        'image_path':image_path
        }
    if nlayer == 3:
        return {
        'num_rows':window,
        'num_cols':window,
        'num_channels':2,
        'batch_size':128,
        'output_channels':[nfilter,2*nfilter,4*nfilter],
        'kernel_shape':[[5,5],[5,5],[5,5]],
        'pool_shape':[[2,2],[2,2],[2,2,]],
        'pool_stride':[[2,2],[2,2],[2,2]],
        'irange':[0.05,0.05,0.05,0.05],
        'max_kernel_norm':[1.9365,1.9365,1.9365],
        'learning_rate':0.003,
        'init_momentum':0.9,
        'weight_decay':[0.00002,0.00002,0.00002,0.00002],
        'n_epoch': 100,
        'train_path':train_path,
        'test_path':test_path,
        'valid_path':valid_path,
        'save_path':save_path,
        'predict_path':predict_path,
        'image_path':image_path
        }
    raise Exception('nlayer can only be 2 or 3!')
def get_default_configure(window=40,nlayer=2,nfilter=64):

    import os
    global __predictDir__, __modelDir__, __datasetDir__, __sarImageDir__
    dataset_dir = '../dataset/sar_hhv_pkl_'+str(window)+'/'
    # 2 generate dataset if not existing
    #if not os.path.exists(__datasetDir__):
    #    extract_patches.prepare_data_window(window)

    predict_path = __predictDir__+'train_with_2010_{}l_{}_{}/'.format(nlayer,window,nfilter)

    if not os.path.exists(predict_path):
        os.makedirs(predict_path)
    days = [
            '20100730',
            '20100806',
            '20100822',
            '20100829',
            '20100909',
            '20100929',
            '20101003',
            '20101006',
            '20101008',
            '20110709',
            '20110710',
            '20110720',
            '20110817',
            '20110725',
            '20110811']
    save_path = 'train_with_2010_{}l_{}_{}.pkl'.format(nlayer,window,nfilter)

    train_path = [dataset_dir+day+'-HHV-8by8-patch.pkl' for day in days[0:13]]
    test_path = [dataset_dir+day+'-HHV-8by8-patch.pkl' for day in [days[13]]]
    valid_path = [dataset_dir+day+'-HHV-8by8-patch.pkl' for day in [days[14]]]

    image_path = [(__sarImageDir__+day+'-HH-8by8-mat.tif',
    __sarImageDir__+day+'-HV-8by8-mat.tif') for day in days]

    if nlayer == 2:
        return {
        'num_rows':window,
        'num_cols':window,
        'num_channels':2,
        'batch_size':128,
        'output_channels':[nfilter,nfilter*2],
        'kernel_shape':[[5,5],[5,5]],
        'pool_shape':[[2,2],[2,2]],
        'pool_stride':[[2,2],[2,2]],
        'irange':[0.05,0.05,0.05],
        'max_kernel_norm':[1.9365,1.9365],
        'learning_rate':0.003,
        'init_momentum':0.9,
        'weight_decay':[0.0005,0.0005,0.0005],
        'n_epoch': 200,
        'train_path':train_path,
        'test_path':test_path,
        'valid_path':valid_path,
        'save_path':save_path,
        'predict_path':predict_path,
        'image_path':image_path
        }
    if nlayer == 3:
        return {
        'num_rows':window,
        'num_cols':window,
        'num_channels':2,
        'batch_size':128,
        'output_channels':[nfilter,2*nfilter,4*nfilter],
        'kernel_shape':[[5,5],[5,5],[5,5]],
        'pool_shape':[[2,2],[2,2],[2,2,]],
        'pool_stride':[[2,2],[2,2],[2,2]],
        'irange':[0.05,0.05,0.05,0.05],
        'max_kernel_norm':[1.9365,1.9365,1.9365],
        'learning_rate':0.003,
        'init_momentum':0.9,
        'weight_decay':[0.00002,0.00002,0.00002,0.00002],
        'n_epoch': 100,
        'train_path':train_path,
        'test_path':test_path,
        'valid_path':valid_path,
        'save_path':save_path,
        'predict_path':predict_path,
        'image_path':image_path
        }
    raise Exception('nlayer can only be 2 or 3!')


def get_default_configure_leave_one_out(window=40,nlayer=2,nfilter=64,
         which_fold=0):
    import os
    global __predictDir__, __modelDir__, __datasetDir__, __sarImageDir__
    dataset_dir = '../dataset/sar_hhv_pkl_'+str(window)+'/'
    # 2 generate dataset if not existing

    predict_path = __predictDir__+'train_with_2010_{}l_{}_{}_leaveoneout/'.format(nlayer,window,nfilter)

    if not os.path.exists(predict_path):
        os.makedirs(predict_path)
    days = [
            '20100730',
            '20100806',
            '20100822',
            '20100829',
            '20100909',
            '20100929',
            '20101003',
            '20101006',
            '20101008',
            '20110709',
            '20110710',
            '20110720',
            '20110817',
            '20110725',
            '20110811']
    save_path = 'train_with_2010_{}l_{}_{}.pkl'.format(nlayer,window,nfilter)
    traindays = [0,1,3,8,9,10,11]
    rest = [2,4,5,6,7,12,13]
    N = len(rest)
    if which_fold>=N:
        raise ValueError('which_fold must < {}'.format(N))
    test_path = [dataset_dir+day+'-HHV-8by8-patch.pkl' for day in [days[rest.pop(which_fold)]]]
    traindays = traindays+rest
    train_path = [dataset_dir+days[day]+'-HHV-8by8-patch.pkl' for day in traindays]
    valid_path = [dataset_dir+day+'-HHV-8by8-patch.pkl' for day in [days[14]]]

    #which_fold = (which_fold+1) % N
    image_path = [(__sarImageDir__+day+'-HH-8by8-mat.tif',
    __sarImageDir__+day+'-HV-8by8-mat.tif') for day in days]

    if nlayer == 2:
        return {
        'num_rows':window,
        'num_cols':window,
        'num_channels':2,
        'batch_size':128,
        'output_channels':[nfilter,nfilter*2],
        'kernel_shape':[[5,5],[5,5]],
        'pool_shape':[[2,2],[2,2]],
        'pool_stride':[[2,2],[2,2]],
        'irange':[0.05,0.05,0.05],
        'max_kernel_norm':[1.9365,1.9365],
        'learning_rate':0.003,
        'init_momentum':0.9,
        'weight_decay':[0.0005,0.0005,0.0005],
        'n_epoch': 500,
        'train_path':train_path,
        'test_path':test_path,
        'valid_path':valid_path,
        'save_path':save_path,
        'predict_path':predict_path,
        'image_path':image_path
        }
    if nlayer == 3:
        return {
        'num_rows':window,
        'num_cols':window,
        'num_channels':2,
        'batch_size':128,
        'output_channels':[nfilter,2*nfilter,4*nfilter],
        'kernel_shape':[[5,5],[5,5],[5,5]],
        'pool_shape':[[2,2],[2,2],[2,2,]],
        'pool_stride':[[2,2],[2,2],[2,2]],
        'irange':[0.05,0.05,0.05,0.05],
        'max_kernel_norm':[1.9365,1.9365,1.9365],
        'learning_rate':0.003,
        'init_momentum':0.9,
        'weight_decay':[0.00002,0.00002,0.00002,0.00002],
        'n_epoch': 100,
        'train_path':train_path,
        'test_path':test_path,
        'valid_path':valid_path,
        'save_path':save_path,
        'predict_path':predict_path,
        'image_path':image_path
        }
    raise Exception('nlayer can only be 2 or 3!')

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

def evaluate_(predictfile,imagefile):
    global __predictDir__, __modelDir__, __datasetDir__, __sarImageDir__

def cnn_ensemble():
    import os
    from datetime import datetime
    print str(datetime.now())
    t0 = time.time()
    kwargs = get_default_configure()
    kwargs['num_rows']=40
    kwargs['num_cols']=40
    import pprint
    pp=pprint.PrettyPrinter(indent=4)
    pp.pprint(kwargs)

    i = -1
    while load_data_balance_under_sample.which_fold != i:
        if i == -1:
            i = load_data_balance_under_sample.which_fold
        kwargs['save_path'] =  str(load_data_balance_under_sample.which_fold)+'.pkl'
        t1 = time.time()
        ann = cnn_train(**kwargs)
        serial.save(kwargs['predict_path']+'f'+kwargs['save_path'],ann,on_overwrite='backup')
        print 'saved to: '+kwargs['save_path']
        print 'Traing done. Take {}h'.format((time.time()-t1)/3600)
        break
    utils.sms_notice('Training finished. Taking {}h in total.'.format((time.time()-t0)/3600))
    print 'Traing done. Take {}h'.format((time.time()-t0)/3600)
    # sum of all predictions
    predict_batch()

def cnn_transform_ensemble():
    import os
    from datetime import datetime
    print str(datetime.now())
    t0 = time.time()
    kwargs = get_default_configure()
    kwargs['num_rows']=41
    kwargs['num_cols']=41
    import pprint
    pp=pprint.PrettyPrinter(indent=4)
    pp.pprint(kwargs)

    i = -1
    while under_sample_water.which_fold != i:
        if i == -1:
            i = under_sample_water.which_fold
        kwargs['save_path'] =  str(under_sample_water.which_fold)+'.pkl'
        t1 = time.time()
        ann = cnn_train_tranformer(**kwargs)
        serial.save(kwargs['predict_path']+'f'+kwargs['save_path'],ann,on_overwrite='backup')
        print 'saved to: '+kwargs['save_path']
        print 'Traing done. Take {}h'.format((time.time()-t1)/3600)
        break
    utils.sms_notice('Training finished. Taking {}h in total.'.format((time.time()-t0)/3600))
    print 'Traing done. Take {}h'.format((time.time()-t0)/3600)
    # sum of all predictions
    predict_batch(predict_path)

def cnn_run():
    import os
    from datetime import datetime
    print str(datetime.now())
    t0 = time.time()
    kwargs = get_default_configure_gsl2014()
    pp=pprint.PrettyPrinter(indent=4)
    pp.pprint(kwargs)
    t1 = time.time()
    ann = cnn_train(**kwargs)
    serial.save(kwargs['predict_path']+'f'+kwargs['save_path'],ann,on_overwrite='backup')
    print 'saved to: '+kwargs['save_path']
    print 'Traing done. Take {}h'.format((time.time()-t1)/3600)
    utils.sms_notice('Training finished. Taking {}h in total.'.format((time.time()-t0)/3600))
    print 'Traing done. Take {}h'.format((time.time()-t0)/3600)

    predict_batch(get_default_configure_gsl2014())
    #evaluate_sets(kwargs['predict_path'])

def cnn_ensemble_leave_one_out():
    import os
    from datetime import datetime
    print str(datetime.now())
    t0 = time.time()
    which_fold = 0
    while 1:
        print which_fold
        try:
            kwargs = get_default_configure_leave_one_out(which_fold=which_fold)
        except:
            break
        kwargs['num_rows']=40
        kwargs['num_cols']=40
        pp=pprint.PrettyPrinter(indent=4)
        pp.pprint(kwargs)
        kwargs['save_path'] =  str(which_fold)+'.pkl'
        t1 = time.time()
        ann = cnn_train(**kwargs)
        serial.save(kwargs['predict_path']+'f'+kwargs['save_path'],ann,on_overwrite='backup')
        print 'saved to: '+kwargs['save_path']
        print 'Traing done. Take {}h'.format((time.time()-t1)/3600)
        which_fold += 1
    utils.sms_notice('Training finished. Taking {}h in total.'.format((time.time()-t0)/3600))
    print 'Traing done. Take {}h'.format((time.time()-t0)/3600)
    # sum of all predictions
    #predict_batch()
    evaluate_sets(kwargs['predict_path'])

def predict_batch(kwargs):
    import os
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    print kwargs
    train_path = kwargs['train_path']
    ann_files = os.listdir(kwargs['predict_path'])
    image_path = kwargs['image_path']
    image_path.reverse()
    for ann_file in ann_files:
        if ann_file.endswith('.pkl'):
            print ann_file
            filename = ann_file.split('.')[0]
            if not os.path.exists(filename):
                os.makedirs(filename)
            for imp in image_path:
                print imp
                day = imp[0][imp[0].rfind('/')+1:]
                day = day[0:day.find('-')]
                output_path = filename+'/'+day+'-ic.tif'
                im = predict_from_file(kwargs['predict_path']+ann_file,
                imp, output_path)
    print "Done!"

def evaluate(ann, patches, cost='None'):
    '''
        evaluate the estimation precision against ground truth, here it is ice charts
        ann: the input ann file or ann object
        patches: the patches to verify on, each patch is a pair of (x,y),
            could be filename or patch object
        cost: is evaluation function, if None, use default
    '''
    patches = (np.transpose(patches[0], (0,2,3,1)),patches[1])
    batch_size = ann.batch_size
    X = ann.get_input_space().make_batch_theano()
    y = ann.fprop(X)
    f = theano.function([X],y)
    #moving window
    m = len(patches)
    im_pred_tmp = np.zeros(shape=(m,ann.get_output_space().dim))
    i = 0
    e = np.zeros((len(patches[0]),1))
    while i<len(patches[0])-len(patches[0])%ann.batch_size:
        e[i:i+ann.batch_size] = f(patches[0][i:i+ann.batch_size])
        i = i+ann.batch_size
    if i < len(patches[0]): e[i:] = f(patches[0][i:])
    e[e>1]=1
    e[e<0]=0
    e = e-patches[1]
    print 'mean error:'+str(np.array(e).mean())
    print 'abs mean error:' + str(np.abs(e).mean())
    print 'std:' + str(np.array(e).std())
    return [np.array(e).mean(), np.abs(e).mean(), np.array(e).std()]

def evaluate_sets_ann(ann,kwargs):
    if type(ann) == str:
        try:
            with open(ann) as f:
                ann = cPickle.load(ann)
        except:
            raise ValueError('The input ann file path is not correct~')
    test = load_data(kwargs['test_path'],kwargs['num_rows'],kwargs['num_cols'],kwargs['num_channels'])
    evaluate(ann,test)
    valid = load_data(kwargs['valid_path'],kwargs['num_rows'],kwargs['num_cols'],kwargs['num_channels'])
    evaluate(ann,valid)
    train = load_data(kwargs['train_path'],kwargs['num_rows'],kwargs['num_cols'],kwargs['num_channels'])
    evaluate(ann,train)

def evaluate_sets(basefolder):
    e_valid = []
    e_train = []
    e_test = []
    which_fold = 0
    test_date = []
    while 1:
        try:
            kwargs = get_default_configure_leave_one_out(which_fold=which_fold)
            #pp=pprint.PrettyPrinter(indent=4)
            #pp.pprint(kwargs)
        except:
            break
        ann_file = which_fold
        print ann_file
        ann_file = basefolder +str(ann_file)+'.pkl'
        #ann_file = '../output/train_with_2010_2l_40_64/original_500/0/0.pkl'
        try:
            with open(ann_file) as f:
                ann_file = cPickle.load(f)
        except:
            raise ValueError('The input ann file path is not correct~')

        test = load_data(kwargs['test_path'],kwargs['num_rows'],kwargs['num_cols'],kwargs['num_channels'])
        e_test.append(evaluate(ann_file,test))
        valid = load_data(kwargs['valid_path'],kwargs['num_rows'],kwargs['num_cols'],kwargs['num_channels'])
        e_valid.append(evaluate(ann_file,valid))
        train = load_data(kwargs['train_path'],kwargs['num_rows'],kwargs['num_cols'],kwargs['num_channels'])
        e_train.append(evaluate(ann_file,train))
        test_date.append(kwargs['test_path'][0].split('/')[-1][0:8])

        which_fold += 1
    e_test = np.asarray(e_test)*100
    e_valid = np.asarray(e_valid)*100
    e_train= np.asarray(e_train)*100
    np.savetxt('e_test.txt',e_test,fmt='%.10f')
    np.savetxt('e_train.txt',e_train,fmt='%.10f')
    np.savetxt('e_valid.txt',e_valid,fmt='%.10f')

    export_latex('e_test.tex',test_date,e_test)
    export_latex('e_valid.tex',test_date,e_valid)
    export_latex('e_train.tex',test_date,e_train)


    tex_file = open('table_errors_leave_one_out.tex','w')
    s = '''\\begin{table}[h]
        \\centering
        \\begin{tabular}{l||lllllllll}
        \\hline
        & $E_{sgn}$ & $E_{L1}$ & $E_{std}$ & $E_{sgn}$ & $E_{L1}$ & $E_{std} \\\\
        \\hline\n'''
    tex_file.write(s)
    for i in range(len(e_train)):
        kwargs = get_default_configure_leave_one_out(which_fold=i)
        s = kwargs['test_path'][0].split('/')[-1][0:8]
        p1 = '\t& %.2f\t& %.2f\t& %.2f' %tuple(e_train[i,:])
        p2 = '\t& %.2f\t& %.2f\t& %.2f' %tuple(e_valid[i,:])
        p3 = '\t& %.2f\t& %.2f\t& %.2f\\\\\n' %tuple(e_test[i,:])
        s = s+p1+p2+p3
        tex_file.write(s)
    f = lambda X: [np.mean(x) for x in np.array(X).transpose()]
    [e_train, e_test, e_valid] = map(f, [e_train, e_test, e_valid])
    s = 'Average'
    p1 = '\t& %.2f\t& %.2f\t& %.2f' %tuple(e_train)
    p2 = '\t& %.2f\t& %.2f\t& %.2f' %tuple(e_valid)
    p3 = '\t& %.2f\t& %.2f\t& %.2f\\\\\n' %tuple(e_test)
    tex_file.write(s)

    s = '''\\end{tabular}
        \\caption{The average error statistics for train, test and validataion datasets.}
        \\label{table:table_errors}
        \\end{table}\n'''
    tex_file.write(s)
    tex_file.close()

def export_latex(fname,days,X):
    # make sure fname end with .tex
    # X is a matrix with each row is the error evaluation for one configuration
    tex_file = open(fname,'w')
    s = '''\\begin{table}[h]
        \\centering
        \\begin{tabular}{l||lll}
        \\hline
        & Date & $E_{sig}$ & $E_{L1}$ & Std. \\\\
        \\hline\n'''
    tex_file.write(s)
    for i in range(len(X)):
        s = '&' + str(days[i])+  ' \t& %.2f\t& %.2f\t& %.2f\\\\\n' % tuple(X[i,:])
        tex_file.write(s)
    s = '''\\end{tabular}
        \\caption{The error statistics for train or test or validataion datasets.}
        \\label{table:table_errors}
        \\end{table}\n'''
    tex_file.write(s)
    tex_file.close()


if __name__ == '__main__':
   cnn_run()
   #cnn_ensemble_leave_one_out()
