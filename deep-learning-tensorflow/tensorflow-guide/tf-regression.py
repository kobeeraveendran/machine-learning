import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

x_data = np.linspace(0.0, 10.0, 1000000)
noise = np.random.randn(len(x_data))

y_true = (0.5 * x_data) + 5 + noise

x_df = pd.DataFrame(data = x_data, columns = ['X data'])
y_df = pd.DataFrame(data = y_true, columns = ['Y'])

my_data = pd.concat([x_df, y_df], axis = 1)

# visualize the data
my_data.sample(n = 250).plot(kind = 'scatter', x = 'X data', y = 'Y')
plt.show()

# prepare batches of data to feed into the network at a time
batch_size = 10

m = tf.Variable(0.69)
b = tf.Variable(0.42)

xph = tf.placeholder(tf.float32, [batch_size])
yph = tf.placeholder(tf.float32, [batch_size])

y_model = m * xph + b

# apply the mean-squared error loss function
error = tf.reduce_sum(tf.square(yph - y_model))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    # initialize variables
    sess.run(init)

    batches = 10000

    input_size = len(x_data)

    for i in range(batches):
        rand_ind = np.random.randint(input_size, size = batch_size)

        dictionary = {xph: x_data[rand_ind], yph: y_true[rand_ind]}

        sess.run(train, feed_dict = dictionary)

    model_m, model_b = sess.run([m, b])

    print("model m: " + str(model_m))
    print("model b: " + str(model_b))

y_hat = x_data * model_m + model_b
my_data.sample(n = 250).plot(kind = 'scatter', x = 'X data', y = 'Y')
plt.plot(x_data, y_hat, color = "red")
plt.show()