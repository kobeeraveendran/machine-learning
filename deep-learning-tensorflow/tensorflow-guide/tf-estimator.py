import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

x_data = np.linspace(0.0, 10.0, 1000000)
y_true = (0.5 * x_data) + np.random.randn(len(x_data))

feat_cols = [ tf.feature_column.numeric_column('x', shape = [1]) ]

estimator = tf.estimator.LinearRegressor(feature_columns = feat_cols)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_true, test_size = 0.3)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

input_function = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train, 
                                batch_size = 8, num_epochs = None, shuffle = True)

train_input_func = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train,
                                batch_size = 8, num_epochs = 1000, shuffle = False)

test_input_func = tf.estimator.inputs.numpy_input_fn({'x': x_test}, y_test,
                                batch_size = 8, num_epochs = 1000, shuffle = False)

estimator.train(input_fn = input_function, steps = 1000)

train_metrics = estimator.evaluate(input_fn = train_input_func, steps = 1000)
print(train_metrics)

test_metrics = estimator.evaluate(input_fn = test_input_func, steps = 1000)
print(test_metrics)

new_data = np.linspace(0, 10, 10)
input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x':new_data}, shuffle = False)

print(list(estimator.predict(input_fn = input_fn_predict)))

predictions = []

for pred in estimator.predict(input_fn = input_fn_predict):
    predictions.append(pred['predictions'])

print(predictions)
