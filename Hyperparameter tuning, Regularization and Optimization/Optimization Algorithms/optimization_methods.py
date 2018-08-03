import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from opt_utils import load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation
from opt_utils import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset
from testCases import *

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def update_parameters_with_gd(parameters, grads, learning_rate):

    L = len(parameters) // 2

    for l in range(L):
        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)]
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]

    return parameters

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):

    np.random.seed(seed)

    m = X.shape[1]
    mini_batches = []

    # synchronously shuffle training and testing sets
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    num_complete_minibatches = math.floor(m / mini_batch_size)

    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k + 1) * mini_batch_size]

        mini_batches.append((mini_batch_X, mini_batch_Y))

    if m % mini_batch_size != 0:
        remainder = m % mini_batch_size
        mini_batch_X = shuffled_X[:, m - remainder : m]
        mini_batch_Y = shuffled_Y[:, m - remainder : m]

        mini_batches.append((mini_batch_X, mini_batch_Y))

    return mini_batches

# check with GD
parameters, grads, learning_rate = update_parameters_with_gd_test_case()

parameters = update_parameters_with_gd(parameters, grads, learning_rate)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

# check mini bathces

X_assess, Y_assess, mini_batch_size = random_mini_batches_test_case()
mini_batches = random_mini_batches(X_assess, Y_assess, mini_batch_size)

print("shape of the first mini_batch_X: " + str(mini_batches[0][0].shape))
print("shape of the second mini_batch_X: " + str(mini_batches[1][0].shape))
print("shape of the third mini_batch_X: " + str(mini_batches[3][0].shape))
print("shape of the first mini_batch_Y: " + str(mini_batches[0][1].shape))
print("shape of the second mini_batch_Y: " + str(mini_batches[0][2].shape))
print("shape of the third mini_batch_Y: " + str(mini_batches[0][3]))
print("mini batch sanity check: " + str(mini_batches[0][0][0][0 : 3]))