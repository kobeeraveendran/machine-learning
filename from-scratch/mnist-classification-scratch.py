# Kobee Raveendran
# neural networks for MNIST handwritten-digit classification from scratch
# NOTE: this uses a (slower) iterative approach rather than using ndarrays
# I'll attempt a vectorized approach to include later

import numpy as np
import random

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # since the input layer doesn't have weights or biases,
        # start from layer 2
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-1 * z))

def feedforward(self, a):
    for b, w in zip(self.biases, self.weights):
        a = sigmoid(np.dot(w, a) + b)
    
    return a

def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, learning_rate, test_data = None):
    if test_data:
        n_test = len(test_data)

    num_examples = len(training_data)

    for i in range(epochs):
        random.shuffle(training_data)

        # semgents training data into chunks of size mini_batch_size
        # and stores into vector totaling the total number of training examples
        mini_batches = [
            training_data[k:k + mini_batch_size]
            for k in range(0, num_examples, mini_batch_size)
        ]

        # go through each mini batch and update weights and biases
        # using gradient descent and backpropagation
        for mini_batch in mini_batches:
            self.update_mini_batch(mini_batch, learning_rate)

        if test_data:
            print("Epoch {0}: {1} / {2}".format(i, self.evaluate(test_data), n_test))

        else:
            print("Epoch {0} complete".format(i))

def update_mini_batch(self, mini_batch, learning_rate):
    # initialize weights and biases with zeros
    db = [np.zeros(b.shape) for b in self.biases]
    dw = [np.zeros(w.shape) for w in self.weights]

    for x, y in mini_batch:
        # calculate gradients for weights and biases
        bgrads, wgrads = self.backpropagation(x, y)
        # determines how much weights and biases should change
        # using the gradient
        db = [nb + dnb for nb, dnb in zip(db, bgrads)]
        dw = [nw + dnw for nw, dnw in zip(dw, wgrads)]

    mini_batch_size = len(mini_batch)

    # below, nw is nabla w, or the partial derivative w.r.t. W, and
    # nb is nabla b, or the partial derivative w.r.t. B
    # update the weights matrix by essentially doing W - ndW, where n is learning rate
    self.weights = [w - (learning_rate / mini_batch_size) * nw
                    for w, nw in zip(self.weights, dw)]
    # update biases by doing B - ndB, w here n is learning rate
    self.biases = [b - (learning_rate / mini_batch_size) * nb
                    for b, nb in zip(self.biases, db)]