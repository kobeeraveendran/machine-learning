import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

train_x_orig, train_y_orig, test_x_orig, test_y, classes = load_data()

# load a sample image
# index = 10
# plt.imshow(train_ox_orig[index])
# print('y = ' + str(train_y[0, index]) + ". It's a " + classes[train_y[0, index]].decode('utf-8') + ' picture.')

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print('Number of training examples: ' + str(m_train))
print('Number of testing examples: ' + str(m_test))
print('Each image is of size: (' + str(num_px) + '3)')
print('train_x_orig shape: ' + str(train_x_orig.shape))
print('train_y_orig shape: ' + str(train_y_orig.shape))
print('test_x_orig shape: ' + str(test_x_orig.shape))
print('test_y shape: ' + str(test_y.shape))

# the -1 argument makes reshape flatten all remaining dimensions
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

train_x = train_x_flatten / 255
test_x = test_x_flatten / 255

print("train_x's shape: " + str(train_x.shape))
print("test_x's shape: " + str(test_x.shape))

# 2 layer neural network and L-layer neural network
# input is (63, 63, 3) flattened to (12288, 1)

# General Methodology
# 1. init parameters, define hyperparameters
# 2. loop for num_iterations:
#       a. forward prop
#       b. compute costs
#       c. backward prop
#       d. update parameters (using parameters and grads from backprop)
# 3. Use trained parameters to predict labels

# 2 layer neural net

def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(1)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    return parameters

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache

def sigmoid(Z):
    A =  1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache

def relu(Z):
    A = np.maximum(0, Z)
    cache = Z

    return A, cache

def sigmoid_backward(dA, cache):
    Z = cache
    s = sigmoid(Z)
    dZ = dA * s * (1 - s)

    return dZ

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0

    return dZ

def linear_activation_forward(A_prev, W, b, activation):
    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation = 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    cache = (linear_cache, activation_cache)

    return A, cache

def compute_cost(AL, Y):
    m = Y.shape[1]

    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.mulitply(1 - Y, np.log(1 - AL)))
    cost = np.squeeze(cost)

    return cost

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

# L-layer neural network

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros([layer_dims[l], 1])

    return parameters