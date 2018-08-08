import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage

import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import load_dataset, random_mini_batches, convert_to_one_hot, forward_propagation_for_predict, predict

np.random.seed(1)

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# example image
plt.imshow(X_train_orig[6])
plt.show()
print('y = ' + str(np.squeeze(Y_train_orig[:, 6])))

X_train = X_train_orig / 255.0
X_test = X_test_orig / 255.0
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

print('number of training examples: ' + str(X_train.shape[0]))
print('number of testing examples: ' + str(X_test.shape[0]))
print('X_train shape: ' + str(X_train.shape))
print('Y_train shape: ' + str(Y_train.shape))
print('X_test shape: ' + str(X_test.shape))
print('Y_test shape: ' + str(Y_test.shape))

conv_layers = {}

def create_placeholders(n_H0, n_W0, n_C0, n_y):

    X = tf.placeholder(dtype = 'float', shape = (None, n_H0, n_W0, n_C0))
    Y = tf.placeholder(dtype = 'float', shape = (None, n_y))

    return X, Y

X, Y = create_placeholders(64, 64, 3, 6)
print('X = ' + str(X))
print('Y = ' + str(Y))


def initialize_parameters():

    tf.set_random_seed(1)

    W1 = tf.get_variable('W1', shape = (4, 4, 3, 8), initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable('W2', shape = (2, 2, 8, 16), initializer = tf.contrib.layers.xavier_initializer(seed = 0))

    parameters = {'W1': W1, 'W2': W2}

    return parameters

tf.reset_default_graph()

with tf.Session() as sess:
    parameters = initialize_parameters()
    init = tf.global_variables_initializer()
    sess.run(init)

    print('W1 = ' + str(parameters['W1'].eval()[1, 1, 1]))
    print('W2 = ' + str(parameters['W2'].eval()[1, 1, 1]))
    
