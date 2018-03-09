# Kobee Raveendran
# deep learning with TensorFlow on the MNIST digits dataset

# some notes about neural nets:
# the process:
'''
input > weight > hidden layer 1 (activ. function) > weights > hidden layer 2 (activ. function) > weights > output layer
this is 'feed forward'

compare output to actual answer > cost function (cross entropy) > optimization function (optimizer) >
cost minimization (stochastic gradient descent, AdaGrad, AdamOptimizer, etc.)
this is 'backpropogation'

1 pass feed forward + 1 pass backpropogation = 1 epoch
'''
# gets rid of annoying CPU warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

# using the MNIST dataset for handwritten digit classification
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/data/', one_hot = True)

# 10 classes, 0 - 9
'''
according to one_hot:
an output of 0 would be represented as: 0 = [1,0,0,0,0,0,0,0,0,0]
one element is 'hot' or on, while the rest are 'cold' or off
'''

# hidden layer sizes
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10      # since there are only 10 digits
batch_size = 100    # goes through batches of 100 images at a time to feed into the network

# X is input image, y is output digit
# so, the x values will be the dimensions 'squashed' together
# note for the future: try 28 by 28 dimensions later, instead of 0 by 784
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neuralNetworkModel(data):
    # creates a tensor of a ton of random variables fitting the dimensions of the input
    # edited to provide increased accuracy using tf.truncated_normal (about 3% increase to 98%)
    hiddenLayer1 = {'weights':tf.Variable(tf.truncated_normal([784, n_nodes_hl1], stddev = 0.1)),
                    'biases': tf.constant(0.1, shape = [n_nodes_hl1])}
    
    hiddenLayer2 = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl1, n_nodes_hl2], stddev = 0.1)), 
                    'biases': tf.constant(0.1, shape = [n_nodes_hl2])}

    hiddenLayer3 = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl2, n_nodes_hl3], stddev = 0.1)), 
                    'biases': tf.constant(0.1, shape = [n_nodes_hl3])}

    outputLayer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl3, n_classes], stddev = 0.1)), 
                    'biases': tf.constant(0.1, shape = [n_classes])}

    layer1 = tf.add(tf.matmul(data, hiddenLayer1['weights']), hiddenLayer1['biases'])
    layer1 = tf.nn.relu(layer1)

    layer2 = tf.add(tf.matmul(layer1, hiddenLayer2['weights']), hiddenLayer2['biases'])
    layer2 = tf.nn.relu(layer2)

    layer3 = tf.add(tf.matmul(layer2, hiddenLayer3['weights']), hiddenLayer3['biases'])
    layer3 = tf.nn.relu(layer3)

    output = tf.add(tf.matmul(layer3, outputLayer['weights']), outputLayer['biases'])
    
    return output

def trainNeuralNetwork(x):
    prediction = neuralNetworkModel(x)
    y = tf.placeholder('float')
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))

    # default learning rate is 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # cycles of feedforward and backpropogation
    num_epochs = 20

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # training stage
        for epoch in range(num_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch:', epoch, 'completed out of:', num_epochs, 'loss:', epoch_loss)

        # run through model, deteremine accuracy
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}) * 100, '%')

trainNeuralNetwork(x)