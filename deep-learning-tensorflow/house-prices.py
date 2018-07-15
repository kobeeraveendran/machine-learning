import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# stop training after periods of little improvement
early_stop = keras.callbacks.EarlyStopping(patience = 20)

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end = '')

NUM_EPOCHS = 500

# train model, printing a '.' after each epoch
history = model.fit(x = train_data, y = train_labels, 
                    epochs = NUM_EPOCHS, 
                    validation_split = 0.2, verbose = 0, 
                    callbacks = [early_stop, PrintDot()])

# visualize training
def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error (in thousands of dollars)')
    plt.plot(history.epoch, 
             np.array(history.history['mean_absolute_error']), 
             label = 'Training Loss')
    plt.plot(history.epoch, 
             np.array(history.history['val_mean_absolute_error']), 
             label = 'Validation Loss')
    plt.legend()
    plt.ylim([0, 5])
    plt.show()

plot_history(history)

[loss, mae] = model.evaluate(x = test_data, y = test_labels, verbose = 0)

print("\nTesting set Mean Absolute Error: ${:7.2f}".format(mae * 1000))

# make predictions
test_predictions = model.predict(test_data).flatten()
print(test_predictions)