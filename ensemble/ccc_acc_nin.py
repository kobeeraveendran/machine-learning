from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Average, Dropout
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras.datasets import cifar10
import numpy as np
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras.backend as K

config = K.tf.ConfigProto()
config.gpu_options.allow_growth = True
session = K.tf.Session(config = config)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train, num_classes = 10)

print('x_train shape: ', x_train.shape)
print('y_train shape: ', y_train.shape)
print('x_test shape: ', x_test.shape)
print('y_test shape: ', y_test.shape)

# shape of an example (all models use the same input shapes)
input_shape = x_train[0, ...].shape
model_input = Input(shape = input_shape)


# model 1: ConvPool - CNN - C
def conv_pool_cnn(model_input):
    
    x = model_input

    # "Striving for Simplicity: The All Convolutional Net"
    x = Conv2D(96, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
    x = Conv2D(96, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
    x = Conv2D(96, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
    x = MaxPooling2D(pool_size = (3, 3), strides = 2)(x)

    x = Conv2D(192, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
    x = Conv2D(192, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
    x = Conv2D(192, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
    x = MaxPooling2D(pool_size = (3, 3), strides = 2)(x)

    x = Conv2D(192, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
    x = Conv2D(192, kernel_size = (1, 1), activation = 'relu', padding = 'same')(x)
    x = Conv2D(10, kernel_size = (1, 1))(x)

    x = GlobalAveragePooling2D()(x)
    x = Activation(activation = 'softmax')(x)

    model = Model(model_input, x, name = 'conv_pool_cnn')

    return model

conv_pool_cnn_model = conv_pool_cnn(model_input)

def compile_and_train(model, num_epochs):
    model.compile(loss = categorical_crossentropy, optimizer = Adam(), metrics = ['accuracy'])

    filepath = 'weights/' + model.name + '.{epoch:02d}-{loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 0, save_weights_only = True, save_best_only = True, mode = 'auto', period = 1)

    tensor_board = TensorBoard(log_dir = 'logs/', histogram_freq = 0, batch_size = 32)

    history = model.fit(x_train, y_train, batch_size = 32, epochs = num_epochs, verbose = 1, callbacks = [checkpoint, tensor_board], validation_split = 0.2)

    return history

#start1 = time.time()
#_ = compile_and_train(conv_pool_cnn_model, 20)
#end1 = time.time()

#print('training time for ConvPool - CNN - C: {} s ({} mins.)'.format(end1 - start1, (end1 - start1) / 60.0))

def evaluate_error(model):

    pred = model.predict(x_test, batch_size = 32)
    pred = np.argmax(pred, axis = 1)
    pred = np.expand_dims(pred, axis = 1)
    error = np.sum(np.not_equal(pred, y_test)) / y_test.shape[0]

    return error

#print('error for ConvPool - CNN: ', evaluate_error(conv_pool_cnn_model))

# model 2: ALL - CNN - C
def all_cnn(model_input):

    x = model_input
    x = Conv2D(96, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
    x = Conv2D(96, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
    x = Conv2D(96, kernel_size = (3, 3), strides = 2, activation = 'relu', padding = 'same')(x)

    x = Conv2D(192, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
    x = Conv2D(192, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
    x = Conv2D(192, kernel_size = (3, 3), strides = 2, activation = 'relu', padding = 'same')(x)

    x = Conv2D(192, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
    x = Conv2D(192, kernel_size = (1, 1), activation = 'relu')(x)
    x = Conv2D(10, kernel_size = (1, 1))(x)

    x = GlobalAveragePooling2D()(x)
    x = Activation(activation = 'softmax')(x)

    model = Model(model_input, x, name = 'all_cnn')

    return model

all_cnn_model = all_cnn(model_input)

start2 = time.time()
_ = compile_and_train(all_cnn_model, 20)
end2 = time.time()

print('training time for All - CNN - C: {} s ({} mins.)'.format(end2 - start2, (end2 - start2) / 60.0))

print('error for All - CNN: ', evaluate_error(all_cnn_model))

def nin_cnn(model_input):

    # block 1
    x = model_input
    x = Conv2D(32, kernel_size = (5, 5), activation = 'relu', padding = 'valid')(x)
    x = Conv2D(32, kernel_size = (1, 1), activation = 'relu')(x)
    x = Conv2D(32, kernel_size = (1, 1), activation = 'relu')(x)
    x = MaxPooling2D(pool_size = (2, 2))(x)
    x = Dropout(0.5)(x)

    # block 2
    x = Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'valid')(x)
    x = Conv2D(64, kernel_size = (1, 1), activation = 'relu')(x)
    x = Conv2D(64, kernel_size = (1, 1), activation = 'relu')(x)
    x = MaxPooling2D(kernel_size = (2, 2))(x)
    x = Dropout(0.5)(x)

    # block 3
    x = Conv2D(128, (3, 3), activation = 'relu', padding = 'valid')(x)
    x = Conv2D(32, kernel_size = (1, 1), activation ='relu')(x)
    x = Conv2D(10, kernel_size = (1, 1))(x)

    x = GlobalAveragePooling2D()(x)
    x = Activation(activation = 'softmax')(x)

    model = Model(model_input, x, name = 'nin_cnn')

    return model

nin_cnn_model = nin_cnn(model_input)

start3 = time.time()
_ = compile_and_train(nin_cnn_model, 20)
end3 = time.time()

print('training time for NIN - CNN: {} s ({} mins.)'.format(end3 - start3, (end3 - start3) / 60.0))

print('error for NIN - CNN: ', evaluate_error(nin_cnn_model))