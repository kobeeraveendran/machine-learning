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
    if check_type == 'vector':
        np.testing.assert_array_almost_equal(actual, desired, err_msg = 'Failed.')
        print('Passed.')

    elif check_type == 'scalar':
        np.testing.assert_almost_equal(actual, desired, err_msg = 'Failed.')
        print('Passed.')

    elif check_type == 'shape':
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

autocheck(gradients['dWaa'][1][2], 10.0, 'scalar')
autocheck(gradients['dWax'][3][1], -10.0, 'scalar')
autocheck(gradients['dWya'][1][2], 0.29713815361, 'scalar')
autocheck(gradients['db'][4], [10.0])
autocheck(gradients['dby'][1], [8.45833407])