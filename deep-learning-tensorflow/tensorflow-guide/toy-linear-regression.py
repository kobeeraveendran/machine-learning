import tensorflow as tf
import random

# adapted from Jacob Buckman's blog

m = tf.get_variable('m', [], initializer = tf.constant_initializer(0.0))
b = tf.get_variable('b', [], initializer = tf.constant_initializer(0.0))
init = tf.global_variables_initializer()

input_placeholder = tf.placeholder(dtype = tf.float32)
output_placeholder = tf.placeholder(dtype = tf.float32)

x = input_placeholder
y = output_placeholder
y_pred = m * x + b

loss = tf.square(y - y_pred)

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-3)
train_optimizer = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(init)

    true_m = random.random()
    true_b = random.random()

    for update_i in range(100000):
        input_data = random.random()
        output_data = true_m * input_data + true_b

        _loss, _ = sess.run([loss, train_optimizer], feed_dict = {input_placeholder: input_data, output_placeholder: output_data})

        print('Iter {} Loss: {}'.format(update_i, _loss))


    print('True params: m = {}, b = {}'.format(true_m, true_b))
    
    learned_params = sess.run([m, b])
    print('Learned params: m = {}, b = {}'.format(learned_params[0], learned_params[1]))

# results:
# true params: m = 0.050988..., b = 0.097086...
# learned params: m = 0.050991..., b = 0.097084...
