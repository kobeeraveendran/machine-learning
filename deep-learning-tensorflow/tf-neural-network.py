# Kobee Raveendran
# Simple linear regression example with tensorflow

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.random.seed(101)
tf.set_random_seed(101)

# simple regression example

x_data = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
y_label = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)

plt.scatter(x_data, y_label)
plt.show()

m = tf.Variable(0.58)
b = tf.Variable(0.69)

error = 0

for x, y  in zip(x_data, y_label):
    y_hat = m * x + b

    # want to minimize the error
    error += (y - y_hat) ** 2

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    training_steps = 100

    for i in range(training_steps):
        sess.run(train)

    final_slope, final_intercept = sess.run([m, b])

x_test = np.linspace(-1, 11, 10)

y_pred_plot = final_slope * x_test + final_intercept

plt.plot(x_test, y_pred_plot, "r")
plt.scatter(x_data, y_label)

plt.show()