import numpy as np
from utils import *
import random

# preprocessing
data = open('dinos.txt', 'r').read()
data = data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('There are {} total characters and {} total unique characters in the dataset.'.format(data_size, vocab_size))

char_to_ix = { ch:i for i, ch in enumerate(sorted(chars)) }
ix_to_char = { i:ch for i, ch in enumerate(sorted(chars)) }

print(ix_to_char)

# (not part of notebook) personal function used to make automatic testing easier
def autocheck(actual, desired, check_type = 'vector'):
    if type(actual) == np.ndarray or check_type == 'vector':
        np.testing.assert_array_almost_equal(actual, desired, err_msg = 'Failed.')
        print('Passed.')

    elif np.isscalar(actual) or check_type == 'scalar':
        np.testing.assert_almost_equal(actual, desired, err_msg = 'Failed.')
        print('Passed.')

    elif type(actual) is tuple:
        print('Passed.') if actual == desired else print('Failed.')


def clip(gradients, maxValue):

    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']

    for gradient in [dWax, dWaa, dWya, db, dby]:
        np.clip(gradient, a_min = -maxValue, a_max = maxValue, out = gradient)

    gradients = {'dWaa': dWaa, 'dWax': dWax, 'dWya': dWya, 'db': db, 'dby': dby}

    return gradients

np.random.seed(3)
dWax = np.random.randn(5, 3) * 10
dWaa = np.random.randn(5, 5) * 10
dWya = np.random.randn(2, 5) * 10
db = np.random.randn(5, 1 ) * 10
dby = np.random.randn(2, 1) * 10

gradients = {'dWax': dWax, 'dWaa': dWaa, 'dWya': dWya, 'db': db, 'dby': dby}
gradients = clip(gradients, 10)

print('\n\nGradient clipping check:')

autocheck(gradients['dWaa'][1][2], 10.0)
autocheck(gradients['dWax'][3][1], -10.0)
autocheck(gradients['dWya'][1][2], 0.29713815361)
autocheck(gradients['db'][4], [10.0])
autocheck(gradients['dby'][1], [8.45833407])


# sampling function
def sample(parameters, char_to_ix, seed):
    """
    Sample a sequence of characters according to a sequence of probability distributions output of the RNN

    Arguments:
    parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b. 
    char_to_ix -- python dictionary mapping each character to an index.
    seed -- used for grading purposes. Do not worry about it.

    Returns:
    indices -- a list of length n containing the indices of the sampled characters.
    """
    
    # Retrieve parameters and relevant shapes from "parameters" dictionary
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]
    
    x = np.zeros((vocab_size, 1))
    
    a_prev = np.zeros((n_a, 1))
    
    indices = []
    
    idx = -1 
    
    counter = 0
    newline_character = char_to_ix['\n']
    
    while (idx != newline_character and counter != 50):
        
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
        z = np.dot(Wya, a) + by
        y = softmax(z)

        np.random.seed(counter+seed) 
        
        idx = np.random.choice(list(range(vocab_size)), p = y.ravel())

        indices.append(idx)
        
        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        
        a_prev = a
        
        seed += 1
        counter +=1
        

    if (counter == 50):
        indices.append(char_to_ix['\n'])
    
    return indices

# sampling function check
np.random.seed(2)
_, n_a = 20, 100
Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}

indices = sample(parameters, char_to_ix, 0)

#print('\n\nSampling function check:')
#print('Sampling indices: ', end = '')
#autocheck(indices, [12, 17, 24, 14, 13, 9, 10, 22, 24, 6, 13, 11, 12, 6, 21, 15, 21, 14, 3, 2, 1, 21, 18, 24, 
#7, 25, 6, 25, 18, 10, 16, 2, 3, 8, 15, 12, 11, 7, 1, 12, 10, 2, 7, 7, 11, 5, 6, 12, 25, 0, 0])

#print('List of sampled characters: ', end = '')
#autocheck([ix_to_char[i] for i in indices], ['l', 'q', 'x', 'n', 'm', 'i', 'j', 'v', 'x', 'f', 'm', 'k', 'l', 'f', 'u', 'o', 
#'u', 'n', 'c', 'b', 'a', 'u', 'r', 'x', 'g', 'y', 'f', 'y', 'r', 'j', 'p', 'b', 'c', 'h', 'o', 
#'l', 'k', 'g', 'a', 'l', 'j', 'b', 'g', 'g', 'k', 'e', 'f', 'l', 'y', '\n', '\n'])


# optimization
def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):

    # forward propagation
    loss, cache = rnn_forward(X, Y, a_prev, parameters)

    # backward propagation
    gradients, a = rnn_backward(X, Y, parameters, cache)

    # gradient clipping
    gradients = clip(gradients, maxValue = 5)

    parameters = update_parameters(parameters, gradients, lr = learning_rate)

    return loss, gradients, a[len(X) - 1]


# optimizer check
np.random.seed(1)
vocab_size, n_a = 27, 100
a_prev = np.random.randn(n_a, 1)
Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
X = [12,3,5,11,22,3]
Y = [4,14,11,22,25, 26]

loss, gradients, a_last = optimize(X, Y, a_prev, parameters)

print('\n\nOptimizer check:')

autocheck(loss, 126.503975722)
autocheck(gradients['dWaa'][1][2], 0.194709315347)
autocheck(np.argmax(gradients['dWax']), 93)
autocheck(gradients['dWya'][1][2], -0.007773876032)
autocheck(gradients['db'][4], [-0.06809825])
autocheck(gradients['dby'][1], [0.01538192])
autocheck(a_last[4], [-1.0])
