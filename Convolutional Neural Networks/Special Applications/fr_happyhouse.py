import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K

K.set_image_data_format('channels_first')

import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd

from fr_utils import *
from inception_blocks_v2 import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.set_printoptions(threshold = np.nan)

FRmodel = faceRecoModel(input_shape = (3, 96, 96))
print('Total parameters = ', FRmodel.count_params())


def triplet_loss(y_true, y_pred, alpha = 0.2):

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    pos_dist = tf.square(tf.linalg.norm(tf.subtract(anchor, positive), axis = -1))
    neg_dist = tf.square(tf.linalg.norm(tf.subtract(anchor, negative), axis = -1))
    
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    
    loss = tf.reduce_sum(tf.maximum(0.0, basic_loss))

    return loss

# check triplet loss
with tf.Session() as test:
    tf.set_random_seed(1)
    y_true = (None, None, None)
    y_pred = (tf.random_normal([3, 128], mean = 6, stddev = 0.1, seed = 1), 
              tf.random_normal([3, 128], mean = 1, stddev = 1, seed = 1), 
              tf.random_normal([3, 128], mean = 3, stddev = 4, seed = 1))

    loss = triplet_loss(y_true, y_pred)
    print('loss = ', loss.eval())

