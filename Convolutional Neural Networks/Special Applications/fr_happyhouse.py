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

# compile model and load pre-trained weights
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)

database = {}

database['danielle'] = img_to_encoding('images/danielle.png', FRmodel)
database['younes'] = img_to_encoding('images/younes.jpg', FRmodel)
database['tian'] = img_to_encoding('images/tian.jpg', FRmodel)
database['andrew'] = img_to_encoding('images/andrew.jpg', FRmodel)
database['kian'] = img_to_encoding('images/kian.jpg', FRmodel)
database['dan'] = img_to_encoding('images/dan.jpg', FRmodel)
database['sebastiano'] = img_to_encoding('images/sebastiano.jpg', FRmodel)
database['bertrand'] = img_to_encoding('images/bertrand.jpg', FRmodel)
database['kevin'] = img_to_encoding('images/kevin.jpg', FRmodel)
database['felix'] = img_to_encoding('images/felix.jpg', FRmodel)
database['benoit'] = img_to_encoding('images/benoit.jpg', FRmodel)
database['arnaud'] = img_to_encoding('images/arnaud.jpg', FRmodel)


def verify(image_path, identity, database, model):

    encoding = img_to_encoding(image_path, model)

    dist = np.linalg.norm(database[identity] - encoding)

    if dist > 0.7:
        print("Faces match, access granted.")
        door_open = True
    else:
        print("No match found, access denied.")
        door_open = False

    return dist, door_open

verify('images/camera_0.jpg', 'younes', database, FRmodel)

verify('images/camera_2.jpg', 'kian', database, FRmodel)


def face_recognition(image_path, database, model):

    encoding = img_to_encoding(image_path, model)

    min_dist = 100

    for (name, db_encoding) in database.items():

        dist = np.linalg.norm(db_encoding - encoding)

        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.7:
        print('No match found, access denied.')

    else:
        print(str(identity) + ' identified. Access granted.')

    return min_dist, identity

# test face rec
face_recognition('images/camera_2.jpg', database, FRmodel)