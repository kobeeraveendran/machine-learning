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

def compute_cost(AL, Y):
    '''
    Arguments:
        AL = probability vector of predictions w/ shape (1, num_examples)
        Y = vector of binary answers

    '''
    m = Y.shape[1]

    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
    cost = np.squeeze(cost)

    return cost

def linear_backward(dZ, cache):
    '''
    Arguments:
        dZ = gradient of cost w.r.t. the linear output
        cache = cache of the values A_prev, W, and b from forward propagation stage
    '''

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev)
    db = (1 / m) * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    '''
    Arguments:
        dA = post-activation gradient for current layer
        cache = cache containing linear_cache, activation_cache
        activation = determines which activation function to use (sigmoid or relu)
    '''
    linear_cache, activation_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    '''
    Arguments:
        AL = probability vector from forward prop (containing the guesses of the NN)
        Y = binary vector of correct answers (classifications)
        caches = list of caches from forward prop (linear_activation_forward() with relu [caches[l], 
                 where l is between 0 and L - 1 (all layers except the last one)], and linear_activation_forward() 
                 with sigmoid [caches[L - 1]])
    '''

    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(y, AL) = np.divide(1 - Y, 1 - AL)) # derivative of cost w.r.t. AL
    current_cache = caches[L - 1]
    grads['dA' + str(L)], grads['dW' + str(L)] = linear_activation_backward(dAL, current_cache, activation = 'sigmoid')

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, activation = 'relu')
        grads['dA' + str(l + 1)] = dA_prev_temp
        grads['dW' + str(l + 1)] = dW_temp
        grads['db' + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    '''
    Arguments:
    parameters = dictionary of parameters
    grads = dictionary of gradients used to update parameters
    '''

    L = len(parameters)

    for l in range(L):
        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * parameters['dW' + str(l + 1)]
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * parameters['db' + str(l + 1)]

    return parameters