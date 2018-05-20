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

def initialize_parameters_deep(layer_dims):
    '''
    Arguments:
        layer_dims = array of dimensions for each layer in the network
    '''

    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros([layer_dims[l], 1])

    return parameters

def linear_forward(A, W, b):
    '''
    Arguments:
        A = results from the activation functions of the previous layer
        W = weights w/ shape (curr layer size, prev layer size)
        b = biases w/ shape (curr layer size, 1)
    '''
    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    '''
    Arguments:
        A_prev = previous layer's activation function output
        W = weights matrix w/ shape (curr_layer_size, prev_layer_size)
        b = biases matrix w shape (curr_layer_size, 1)
        activation = which activation function to use, in string format (i.e. 'sigmoid', 'relu')
    '''

    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    '''
    Arguments:
        X = data in a numpy array w/ shape (input size, num_examples)
        parameters = output of the initializer method
    '''

    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        A, cache = linear_activation_forward(A_prev, W, b, activation = 'relu')
        caches.append(cache)

    W = parameters['W' + str(L)]
    b = parameters['b' + str(L)]
    AL, cache = linear_activation_forward(A, W, b, activation = 'sigmoid')
    caches.append(cache)

    return AL, caches