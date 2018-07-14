import tensorflow as tf
import numpy as np
from tensorflow import keras

# download dataset
imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)

print('Training data size: ' + str(len(train_data)), 
      '\nTraining label size: ' + str(len(train_labels)), 
      '\nTesting data size: ' + str(len(test_data)), 
      '\nTesting label size: ' + str(len(test_labels)))

word_index = imdb.get_word_index()

word_index = {key : (val + 3) for key, val in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3

num_index = dict([(val, key) for (key, val) in word_index.items()])

def decode_review(text):
    return ' '.join([num_index.get(i, '?') for i in text])

#print(decode_review(train_data[0]))

train_data = keras.preprocessing.sequence.pad_sequences(train_data, 
                                                        value = 0, 
                                                        padding = 'post', 
                                                        maxlen = 256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data, 
                                                       value = 0, 
                                                       padding = 'post', 
                                                       maxlen = 256)

print("New lengths: " + str(len(train_data[0])) + " " + str(len(train_data[1])))

vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation = tf.nn.relu))
model.add(keras.layers.Dense(1, activation = tf.nn.sigmoid))

print(model.summary())