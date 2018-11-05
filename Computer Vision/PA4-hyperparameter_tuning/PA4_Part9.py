'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import backend as K
import matplotlib.pyplot as plt

batch_size = 128
num_classes = 10
epochs = 30

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# custom LeNet-5 model by me

# Maximum pooling version


# learning rate choices
learning_rates = [0.009, 0.0001, 0.01, 0.1, 0.09]

for learning_rate in learning_rates:

    # had to redefine the model at each step because for some reason
    # information seemed to be stored or passed between iterations of the 
    # learning rate loop if it wasn't redefined (sometimes starting accuracies for later iterations
    # were just too good to be true)
    model = Sequential()
    model.add(Conv2D(6, kernel_size = (5, 5), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = 2))
    model.add(Conv2D(16, kernel_size = (5, 5), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = 2))
    model.add(Flatten())
    model.add(Dense(120, activation = 'relu'))
    model.add(Dense(84, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))

    model.compile(loss = keras.losses.categorical_crossentropy, 
                  optimizer = keras.optimizers.SGD(lr = learning_rate), 
                  metrics = ['accuracy'])

    history = model.fit(x_train, y_train, 
                        batch_size = batch_size, 
                        epochs = epochs, 
                        verbose = 1, 
                        validation_data = (x_test, y_test))

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('LeNet-5 Accuracy with LR = ' + str(learning_rate))
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train', 'test'], loc = 'upper left')
    plt.show()
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print('\nLearning rate = ', learning_rate)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])