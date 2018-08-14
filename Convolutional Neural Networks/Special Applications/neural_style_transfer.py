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
plt.show()


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

style_image = scipy.misc.imread('images/monet_800600.jpg')
imshow(style_image)
plt.show()

# Gram matrix computation
def gram_matrix(A):

    GA = tf.matmul(A, tf.transpose(A))

    return GA

tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    A = tf.random_normal([3, 2], mean = 1, stddev = 4)
    GA = gram_matrix(A)

    print('GA = ', GA.eval())


def compute_layer_style_cost(a_S, a_G):

    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_S = tf.transpose(tf.reshape(a_S, shape = (n_H * n_W, n_C)))
    a_G = tf.transpose(tf.reshape(a_G, shape = (n_H * n_W, n_C)))

    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    J_style_layer = (1 / (4 * (n_C ** 2) * (n_H * n_W) ** 2)) * tf.reduce_sum(tf.square(tf.subtract(GS, GG)))

    return J_style_layer

# test case
tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    a_S = tf.random_normal([1, 4, 4, 3], mean = 1, stddev = 4)
    a_G = tf.random_normal([1, 4, 4, 3], mean = 1, stddev = 4)

    J_style_layer = compute_layer_style_cost(a_S, a_G)
    print('J_style_layer = ', J_style_layer.eval())


# layer weights (play around with this to see changes in style)
STYLE_LAYERS = [('conv1_1', 0.2), ('conv2_1', 0.2), ('conv3_1', 0.2), 
                ('conv4_1', 0.2), ('conv5_1', 0.2)]


def compute_style_cost(model, STYLE_LAYERS):

    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]

        a_S = sess.run(out)
        a_G = out

        J_style_layer = compute_layer_style_cost(a_S, a_G)

        J_style += coeff * J_style_layer

    return J_style


# minimize style and content cost
def total_cost(J_content, J_style, alpha = 10, beta = 40):

    J = alpha * J_content + beta * J_style

    return J

tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(3)
    J_content = np.random.randn()
    J_style = np.random.randn()
    
    J = total_cost(J_content, J_style)
    print('J = ', J)