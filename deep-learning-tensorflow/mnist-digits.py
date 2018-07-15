import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

mnist = keras.datasets.mnist
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

# explore data
print("Training data shape: " + str(train_data.shape))
print("Training label shape: " + str(train_labels.shape))
print("Testing data shape: " + str(test_data.shape))
print("Testing label shape: " + str(test_labels.shape))

plt.figure()
plt.imshow(train_data[0])
plt.show()

# reshape and flatten for model compatibility
train_data = np.reshape(train_data, (60000, 784))
test_data = np.reshape(test_data, (10000, 784))

# restrict pixel values
train_data = train_data.astype('float32') / 255.0
test_data = train_data.astype('float32') / 255.0

# define model
model = keras.models.Sequential()
model.add(keras.layers.Dense(512, activation = 'relu', input_shape = (784, )))
model.add(keras.layers.Dense(10, activation = 'softmax'))

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
