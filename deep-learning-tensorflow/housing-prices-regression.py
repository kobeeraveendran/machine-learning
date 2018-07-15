import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

housing = keras.datasets.boston_housing

(train_data, train_labels), (test_data, test_labels) = housing.load_data()

print('Training set shape: {} \nTesting set shape: {} \n'.format(train_data.shape, test_data.shape))
print(train_data[0])

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 
                'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 
                'B', 'LSTAT']

df = pd.DataFrame(train_data, columns = column_names)
print(df.head())

print(train_labels[:10])

# feature normalization
mean = train_data.mean(axis = 0)
std = train_data.std(axis = 0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

# confirm with newly-normalized training/testing sample
print(train_data[0])
print(test_data[0])

# build model
def build_model():
    model = keras.Sequential([keras.layers.Dense(64, activation = tf.nn.relu, input_shape = (train_data.shape[1], )), 
                              keras.layers.Dense(64, activation = tf.nn.relu), 
                              keras.layers.Dense(1)])

    model.compile(loss = 'mse', 
                  optimizer = tf.train.RMSPropOptimizer(learning_rate = 0.001), 
                  metrics = ['mae'])

    return model

model = build_model()
model.summary()