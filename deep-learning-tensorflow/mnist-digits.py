import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras import models

mnist = keras.datasets.mnist
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

# explore data
print("Training data shape: " + str(train_data.shape))
print("Training label shape: " + str(train_labels.shape))
print("Testing data shape: " + str(test_data.shape))
print("Testing label shape: " + str(test_labels.shape))

# restrict pixel values
train_data = train_data.astype('float32') / 255
test_data = train_data.astype('float32') / 255

# categorize labels
train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)

# define model, flatten input
model = models.Sequential([
    layers.Flatten(input_shape = (28, 28)), 
    layers.Dense(512, activation = 'relu'), 
    layers.Dense(10, activation = 'softmax')
])

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(x = train_data, y = train_labels, epochs = 5, batch_size = 128)
