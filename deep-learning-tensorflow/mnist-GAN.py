import tensorflow as tf 
import numpy as np 
import input_data
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import os
import scipy.misc
import scipy

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

mnist = input_data.read_data_sets('MNIST_data/', one_hot = False)

# helper functions

# leaky relu activation function (for discriminator network)
def lrelu(x, leak = 0.2, name = "lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)
    
# github.com/carpedm20/DCGAN-tensorflow
def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def inverse_transform(images):
    return (images + 1.) / 2. 

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h: j * h + h, i * w: i * w + w] = image

    return image

# generator network
def generator(z):
    zP = slim.fully_connected(z, 4 * 4 * 256, normalizer_fn = slim.batch_norm, activation_fn = tf.nn.relu, scope = 'g_project', weights_initializer = initializer)
    zCon = tf.reshape(zP, [-1, 4, 4, 256])
    
    gen1 = slim.convolution2d_transpose(zCon, num_outputs = 64, kernel_size = [5, 5], stride = [2, 2], padding = "SAME", normalizer_fn = slim.batch_norm, activation_fn = tf.nn.relu, scope = 'g_conv1', weights_initializer = initializer)

    gen2 = slim.convolution2d_transpose(gen1, num_outputs = 32, kernel_size = [5, 5], stride = [2, 2], padding = "SAME", normalizer_fn = slim.batch_norm, activation_fn = tf.nn.relu, scope = 'g_conv2', weights_initializer = initializer)

    gen3 = slim.convolution2d_transpose(gen2, num_outputs = 16, kernel_size = [5, 5], stride = [2, 2], padding = "SAME", normalizer_fn = slim.batch_norm, activation_fn = tf.nn.relu, scope = 'g_conv3', weights_initializer = initializer)

    g_out = slim.convolution2d_transpose(gen3, num_outputs = 1, kernel_size = [32, 32], padding = "SAME", biases_initializer = None, activation_fn = tf.nn.tanh, scope = 'g_out', weights_initializer = initializer)

    return g_out


# discriminator network
def discriminator(bottom, reuse = False):
    dis1 = slim.convolution2d(bottom, 16, [4, 4], stride = [2, 2], padding = "SAME", biases_initializer = None, activation_fn = lrelu, reuse = reuse, scope = 'd_conv1', weights_initializer = initializer)

    dis2 = slim.convolution2d(dis1, 32, [4, 4], stride = [2, 2], padding = "SAME", normalizer_fn = slim.batch_norm, activation_fn = lrelu, reuse = reuse, scope = 'd_conv2', weights_initializer = initializer)

    dis3 = slim.convolution2d(dis2, 64, [4, 4], stride = [2, 2], padding = "SAME", normalizer_fn = slim.batch_norm, activation_fn = lrelu, reuse = reuse, scope = 'd_conv3', weights_initializer = initializer)

    d_out = slim.fully_connected(slim.flatten(dis3), 1, activation_fn = tf.nn.sigmoid, reuse = reuse, scope = 'd_out', weights_initializer = initializer)

    return d_out

# put all the pieces together in a model
tf.reset_default_graph()

z_size = 100    # size of z vector for the generator

initializer = tf.truncated_normal_initializer(stddev = 0.02)    # initialize all of the weights in the network

z_in = tf.placeholder(shape = [None, z_size], dtype = tf.float32)   # vector of random numbers

real_in = tf.placeholder(shape = [None, 32, 32, 1], dtype = tf.float32)     # real images

Gz = generator(z_in)    # generate images from the random vector
Dx = discriminator(real_in)     # creates probabilities of being real for the real images
Dg = discriminator(Gz, reuse = True)    # creates probabilities of being real for the generator images

d_loss = -tf.reduce_mean(tf.log(Dx) + tf.log(1. - Dg))      # optimizes the discriminator
g_loss = -tf.reduce_mean(tf.log(Dg))    # optimizes the generator

tvars = tf.trainable_variables()

# below, applies gradient to update the GAN
trainerD = tf.train.AdamOptimizer(learning_rate = 0.0002, beta1 = 0.5)
trainerG = tf.train.AdamOptimizer(learning_rate = 0.0002, beta1 = 0.5)
d_grads = trainerD.compute_gradients(d_loss, tvars[9:])     # update weights for discriminator network
g_grads = trainerG.compute_gradients(g_loss, tvars[0:9])    # update weights for generator network

update_D = trainerD.apply_gradients(d_grads)
update_G = trainerD.apply_gradients(g_grads)

# TRAINING

batch_size = 128    # how many images to supply each iteration
iterations = 500000 # total number of iterations
sample_directory = './figs' # where to save sample images from generator
model_directory = './models' # where to save the trained model to

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session(config= config) as sess:
    sess.run(init)

    for i in range(iterations):
        zs = np.random.uniform(-1.0, 1.0, size = [batch_size, z_size]).astype(np.float32)    # generate random z batch
        xs, _ = mnist.train.next_batch(batch_size)      # draw sample batch from mnist dataset
        xs = (np.reshape(xs, [batch_size, 28, 28, 1]) - 0.5) * 2.0  # transform so that it is between -1 and 1
        xs = np.lib.pad(xs, ((0,0), (2,2), (2,2), (0,0)), 'constant', constant_values = (-1, -1))   # pad images so that they become 32x32
        _, dLoss = sess.run([update_D, d_loss], feed_dict = {z_in:zs, real_in:xs})  # update discriminator
        _, gLoss = sess.run([update_G, g_loss], feed_dict = {z_in:zs}) # updating twice just in case
        _, gLoss = sess.run([update_G, g_loss], feed_dict = {z_in:zs})

        if i % 10 == 0:
            print("Generator loss: " + str(gLoss) + " Discriminator loss: " + str(dLoss))
            z2 = np.random.uniform(-1.0, 1.0, size = [batch_size, z_size]).astype(np.float32)   # generate another z batch
            newZ = sess.run(Gz, feed_dict = {z_in:z2})  # use new z to get sample images from generator

            if not os.path.exists(sample_directory):
                os.makedirs(sample_directory)   # make directory if it doesn't exist
            
            save_images( np.reshape(newZ[0:36], [36, 32, 32]), [6, 6], sample_directory + '/fig' + str(i) + '.png')
            
            if i % 1000 == 0 and i != 0:
                if not os.path.exists(model_directory):
                    os.makedirs(model_directory)
                
                saver.save(sess, model_directory + '/model-' + str(i) + '.cptk')
                print("Saved model.")


# how to use the trained (and saved) network model
sample_directory = './figs' # where to save generated sample images
model_directory = './models'    # where to loead the trained model from
batch_size_sample = 36

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session(config = config) as sess:
    sess.run(init)
    # re-load the model
    print('Loading model...')
    checkpoint = tf.train.get_checkpoint_state(model_directory)
    saver.restore(sess, checkpoint.model_checkpoint_path)

    zs = np.random.uniform(-1.0, 1.0, batch_size = [batch_size_sample, z_size]).astype(np.float32)  # generate random z batch
    newZ = sess.run(Gz, feed_dict = {z_in: z2})
    if not os.path.exists(sample_directory):
        os.makedirs(sample_directory)
    
    save_images(np.reshape(newZ[0:batch_size_sample], [36, 32, 32]), [6, 6], sample_directory + '/fig' + str(i) + '.png')

