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

def clip(gradients, maxValue):

    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']

    for gradient in [dWax, dWaa, dWya, db, dby]:
        np.clip(gradient, a_min = -maxValue, a_max = maxValue, out = gradient)

    gradients = {'dWaa': dWaa, 'dWax': dWax, 'dWya': dWya, 'db': db, 'dby': dby}

    return gradients

    