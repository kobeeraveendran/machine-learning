from __future__ import print_function
import IPython
import sys

import numpy as np
from music21 import *
from grammar import *
from qa import *
from preprocess import *
from music_utils import *
from data_utils import *

from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K

# only in interactive notebook
#IPython.display.Audio('data/30s_seq.mp3')

X, Y, n_values, indices_values = load_music_utils()

print('Shape of X: ', X.shape)
print('Number of training examples: ', X.shape[0])
print('T_x (length of sequence): ', X.shape[1])
print('Total number of unique values: ', n_values)
print('Shape of Y: ', Y.shape)
