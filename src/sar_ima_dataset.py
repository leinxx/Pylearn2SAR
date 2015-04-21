#!/usr/bin/env python
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import numpy as np
import cPickle

def load_data(base_folder, data_file_list):
   
    lines = open(base_folder + data_file_list).readlines()
    X = []
    y = []
    for line in lines:
        with open(base_folder + line.strip()) as f:
            if X == []:
                X, y = cPickle.load(f)
                y = y.reshape(len(y),1)#.astype('float64')
            else:
                ds = cPickle.load(f)  
                X = np.concatenate((X,ds[0]),axis=0)
                y = np.concatenate((y,ds[1].reshape(len(ds[1]),1)),axis=0)
        print 'number of samples loaded from ' + line.strip() + str(X.shape[0])
    return (X, y)

class SarImaDataset(DenseDesignMatrix):

    def __init__(self,which_set = None, base_folder = None, 
                  data_file_list = None,
                  preprocessor = None,
                  fit_preprocessor = False,
                  X = None,
                  y = None):

        if data_file_list != None:
            X, y = load_data(base_folder, data_file_list)
          
        self.__dict__.update(locals())
        super(SarImaDataset,self).__init__(topo_view=X,y=y,axes = ('b', 0, 1,'c'))
        if self.X is not None and preprocessor:
            preprocessor.apply(self, fit_preprocessor)
