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
