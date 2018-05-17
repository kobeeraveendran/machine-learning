#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 20:41:32 2018

@author: Kobee Raveendran
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases_v2 import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def initialize_parameters(n_x, n_h, n_y):
    '''
    Arguments:
        n_x = size of input layer
        n_h = size of hidden layer
        n_y = size of output layer
    '''
    
    np.random.seed(1)
    
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    parameters = {'W1': W1, 
                  'b1': b1, 
                  'W2': W2, 
                  'b2': b2}
    
    return parameters