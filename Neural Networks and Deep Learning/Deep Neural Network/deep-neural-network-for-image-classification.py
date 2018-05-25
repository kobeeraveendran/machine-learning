import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v2 import *

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



def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros([layer_dims[l], 1])

    return parameters