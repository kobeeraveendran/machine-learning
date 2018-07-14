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
