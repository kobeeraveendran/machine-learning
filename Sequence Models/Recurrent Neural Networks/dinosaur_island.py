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
def autocheck(actual, desired):
    if type(actual) == np.ndarray:
        np.testing.assert_array_almost_equal(actual, desired, err_msg = 'Failed.')
        print('Passed.')

    elif np.isscalar(actual):
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

    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]

    x = np.zeros((vocab_size, 1))
    a_prev = np.zeros((n_a, 1))

    indices = []
    idx = -1

    counter = 0
    newline_character = char_to_ix['\n']

    while(idx != newline_character and counter != 50):
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
        z = np.dot(Wya, a_prev) + by
        y = softmax(z)

        np.random.seed(counter + seed)

        idx = np.random.choice(list(range(vocab_size)), p = y.ravel())
        indices.append(idx)

        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        
        a_prev = a

        seed += 1
        counter += 1

    if counter == 50:
        indices.append(char_to_ix['\n'])

    return indices
