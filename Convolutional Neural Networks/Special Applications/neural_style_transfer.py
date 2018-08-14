import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

model = load_vgg_model('pretrained-model/imagenet-vgg-verydeep-19.mat')

# content image
content_image = scipy.misc.imread('images/louvre.jpg')
imshow(content_image)


def compute_content_cost(a_C, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_C_unrolled = tf.reshape(a_C, shape = (n_H * n_W, n_C))
    a_G_unrolled = tf.reshape(a_G, shape = (n_H * n_W, n_C))

    J_content = (1 / (4 * n_H * n_W * n_C)) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))

    return J_content

tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    a_C = tf.random_normal([1, 4, 4, 3], mean = 1, stddev = 4)
    a_G = tf.random_normal([1, 4, 4, 3], mean = 1, stddev = 4)

    J_content = compute_content_cost(a_C, a_G)
    print('J_content = ', J_content.eval())

