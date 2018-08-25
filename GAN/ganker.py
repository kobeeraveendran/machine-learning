import os
import numpy as np
import pandas as pd
from scipy.misc import imread
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape, InputLayer
from keras.regularizers import L1L2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

seed = 128
rng = np.random.RandomState(seed)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train / 255.0

img = x_train[0]

plt.imshow(img, cmap = 'gray')
plt.show()

g_input_shape = 100
d_input_shape = (28, 28)
hidden_1_num_units = 500
hidden_2_num_units = 500
g_output_num_units = 784
d_output_num_units = 1
num_epochs = 25
batch_size = 128

# generator model
model1 = Sequential()
model1.add(InputLayer(input_shape = g_input_shape))
model1.add(Dense(units = hidden_1_num_units, activation = 'relu',  kernel_regularizer = L1L2(l1 = 1e-5, l2 = 1e-5)))
model1.add(Dense(units = hidden_2_num_units, activation = 'relu', kernel_regularizer = L1L2(l1 = 1e-5, l2 = 1e-5)))
model1.add(Dense(units = g_output_num_units, activation = 'sigmoid', kernel_regularizer = L1L2(l1 = 1e-5, l2 = 1e-5)))
model1.add(Reshape(d_input_shape))

# discriminator model
model2 = Sequential()
model2.add(InputLayer(input_shape = d_input_shape))
model2.add(Flatten())
model2.add(Dense(units = hidden_1_num_units, activation = 'relu', kernel_regularizer = L1L2(1e-5, 1e-5)))
model2.add(Dense(units = hidden_2_num_units, activation = 'relu', kernel_regularizer = L1L2(1e-5, 1e-5)))
model2.add(Dense(units = d_output_num_units, activation = 'sigmoid', kernel_regularizer = L1L2(1e-5, 1e-5)))
