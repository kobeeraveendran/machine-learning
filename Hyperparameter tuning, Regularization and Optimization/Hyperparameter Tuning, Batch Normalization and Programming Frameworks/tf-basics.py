import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

np.random.seed(1)

# computing loss on a single training example
y_hat = tf.constant(36, name = 'y_hat')
y = tf.constant(39, name = 'y')

loss = tf.Variable((y - y_hat) ** 2, name = 'loss')

init = tf.global_variables_initializer()

with tf.Session as sess:
    sess.run(init)
    print(sess.run(loss))

a = tf.constant(2)
b = tf.constant(10)
c = tf.multiply(a, b)
print(c)

sess = tf.Session()
print(sess.run(c))

x = tf.placeholder(dtype = tf.int64, name = 'x')
print(sess.run(2 * x, feed_dict = {x: 3}))
sess.close()

# linear function, Y = WX + b
def linear_function():

    np.random.seed(1)

    X = tf.constant(np.random.randn(3, 1), name = 'X')
    W = tf.constant(np.random.randn(4, 3), name = 'W')
    b = tf.constant(np.random.randn(4, 1), name = 'b')
    Y = tf.add(tf.matmul(W, X), b)

    # session creation
    sess = tf.Session()
    result = sess.run(Y)

    sess.close()

    return result

def sigmoid(z):

    x = tf.placeholder(dtype = tf.float32, name = 'x')

    sigmoid = tf.sigmoid(x)

    with tf.Session() as sess:
        result = sess.run(sigmoid, feed_dict = {x: z})

    return result

def cost(logits, labels):

    z = tf.placeholder(dtype = tf.float32, name = 'z')
    y = tf.placeholder(dtype = tf.float32, name = 'y')

    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = z, labels = y)

    sess = tf.Session()

    cost = sess.run(cost, feed_dict = {z: logits, y: labels})

    sess.close()

    return cost


# sigmoid check
print('sigmoid(0) = ' + str(sigmoid(0)))
print('sigmoid(12) = ' + str(sigmoid(12)))

# cost function check
logits = sigmoid(np.array([0.2, 0.4, 0.7, 0.9]))
cost = cost(logits, np.array([0, 0, 1, 1]))
print('cost = ' + str(cost))