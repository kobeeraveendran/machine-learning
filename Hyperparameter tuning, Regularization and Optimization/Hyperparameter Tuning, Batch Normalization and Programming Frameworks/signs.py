import math
import numpy as np
import h5py
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

np.random.seed(1)

# load dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# visualize dataset
index = 0
plt.imshow(X_train_orig[index])
plt.show()
print('y = ' + str(np.squeeze(Y_train_orig[:, index])))

# flatten train and test images into 1-D vectors
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

# normalize image vectors to grayscale
X_train = X_train_flatten / 255.0
X_test = X_test_flatten / 255.0

# encode labels as one-hot matrix
Y_train = convert_to_one_hot(Y_train_orig, C = 6)
Y_test = convert_to_one_hot(Y_test_orig, C = 6)

print('number of training examples: ' + str(X_train.shape[1]))
print('number of test examples: ' + str(X_test.shape[1]))
print('X_train shape: ' + str(X_train.shape))
print('Y_train shape: ' + str(Y_train.shape))
print('X_test shape: ' + str(X_test.shape))
print('Y_test shape: ' + str(Y_test.shape))

def create_placeholders(n_x, n_y):
    X = tf.placeholder(dtype = 'float', shape = (n_x, None))
    Y = tf.placeholder(dtype = 'float', shape = (n_y, None))

    return X, Y

def initialize_parameters():
    tf.set_random_seed(1)

    W1 = tf.get_variable(name = "W1", shape = (25, 12288), initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable(name = "b1", shape = (25, 1), initializer = tf.zeros_initializer())
    W2 = tf.get_variable(name = "W2", shape = (12, 25), initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable(name = "b2", shape = (12, 1), initializer = tf.zeros_initializer())
    W3 = tf.get_variable(name = "W3", shape = (6, 12), initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable(name = "b3", shape = (6, 1), initializer = tf.zeros_initializer())

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3
    }

    return parameters

tf.reset_default_graph()
with tf.Session() as sess:
    parameters = initialize_parameters()
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

def forward_propagation(X, parameters):

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    
    return Z3

