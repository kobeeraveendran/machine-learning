import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers

from keras.datasets import imdb

# imdb.load_data returns two tuples: (x_train, y_train) and (x_test, y_test)
# x_train and x_test are lists of sequences (lists of integer indexes)
# y_train and y_test are lists of integer labels (1 or 0) specifying whether the review was good or bad
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words = 10000)
data = np.concatenate((training_data, testing_data), axis = 0)
targets = np.concatenate((training_targets, testing_targets), axis = 0)

print("Categories: ", np.unique(targets))
print("Number of unique words: ", len(np.unique(np.hstack(data))))

length = [len(i) for i in data]
print("Average review length: ", np.mean(length))
print("Standard deviation: ", round(np.std(length)))

print("Labels: ", targets[0])
print(data[0])