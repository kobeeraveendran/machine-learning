import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt

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
