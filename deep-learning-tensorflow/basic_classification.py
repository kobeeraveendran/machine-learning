import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# output classes:
# 0 : t-shirt/top
# 1 : trouser
# 2 : pullover
# 3 : dress
# 4 : coat
# 5 : sandal
# 6 : shirt
# 7 : sneaker
# 8 : bag
# 9 : ankle boot
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 
               'Dress', 'Coat', 'Sandal', 'Shirt', 
               'Sneaker', 'Bag', 'Ankle boot']

# dataset exploration
print("training shape: " + str(train_images.shape), 
      "\ntraining labels: " + str(len(train_labels)), 
      "\ntesting shape: " + str(test_images.shape), 
      "\ntesting labels: " + str(len(test_labels)))

plt.figure()
plt.imshow(train_images[32])
plt.colorbar()
plt.gca().grid(False)
#plt.show()

# squeeze pixel value range from 0, 255 to 0, 1
train_images = np.array(train_images) / 255.0
test_images = np.array(test_images) / 255.0

for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(train_images[i], cmap = 'Greys')
    plt.xlabel(class_names[train_labels[i]])

plt.show()

# define the model
# l1 - flatten from 28 x 28 to 784, 1
# l2, l3 - fully connected layers; l3 contains output probabilities
model = keras.Sequential([keras.layers.Flatten(input_shape = (28, 28)), 
                          keras.layers.Dense(128, activation = tf.nn.relu), 
                          keras.layers.Dense(10, activation = tf.nn.softmax)])

