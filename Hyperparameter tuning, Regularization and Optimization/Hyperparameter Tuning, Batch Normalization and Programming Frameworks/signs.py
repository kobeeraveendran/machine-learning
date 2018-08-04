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