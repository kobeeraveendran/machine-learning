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

