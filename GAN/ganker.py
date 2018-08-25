import os
import numpy as np
import pandas as pd
from scipy.misc import imread
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, LeakyReLU, Dropout, Activation
from keras.regularizers import L1L2

from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

d_model = Sequential()
dropout = 0.4

input_shape = (28, 28, 1)

d_model.add(Conv2D(64, kernel_size = 5, strides = 2, input_shape = input_shape, padding = 'same', activation = LeakyReLU(alpha = 0.2)))
d_model.add(Dropout(dropout))

d_model.add(Conv2D(128, kernel_size = 5, strides = 2, padding = 'same', activation = LeakyReLU(alpha = 0.2)))
d_model.add(Dropout(dropout))

d_model.add(Conv2D(256, kernel_size = 5, strides = 2, padding = 'same', activation = LeakyReLU(alpha = 0.2)))
d_model.add(Dropout(dropout))

d_model.add(Conv2D(512, kernel_size = 5, strides = 2, padding = 'same', activation = LeakyReLU(alpha = 0.2)))
d_model.add(Dropout(dropout))

d_model.add(Flatten())
d_model.add(Dense(units = 1))
d_model.add(Activation(activation = 'sigmoid'))

d_model.summary()