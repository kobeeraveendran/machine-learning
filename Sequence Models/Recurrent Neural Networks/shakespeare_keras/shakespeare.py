from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io

path = get_file('nietzsche.txt', origin = 'https://s3.amazonaws.com/text-datasets/nietzsche.txt')

with io.open(path, encoding = 'utf-8') as f:
    text = f.read().lower()

print('corpus length: ', len(text))

chars = sorted(list(set(text)))
print('total chars: ', len(chars))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((c, i) for i, c in enumerate(chars))

maxlen = 40
step = 3
sentences = []
