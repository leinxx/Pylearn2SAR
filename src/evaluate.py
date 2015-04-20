#!/usr/bin/env python
import sys
import os
import cPickle
import theano
import theano.tensor as T
from pylearn2.models import mlp
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import time
from pylearn2.utils import serial
import numpy as np
import cPickle
import math
import operator
import Image


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

def evaluate_sets(ann_file, base_dir, output_dir):
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

    if len(sys.argv) != 4:
        print '''Usage: ./evaluate.py ann_file base_dir out_dir
          ann_file: the file path of the ann file
          base_dir: the directory of the training data (pkl files)
          out_dir: the output directory of the evaluation results
          '''
    else:
      evaluate_sets(sys.argv[1], sys.argv[2], sys.argv[3])
