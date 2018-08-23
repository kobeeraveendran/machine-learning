from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Average, Dropout
from keras.utils import to_categorical
from keras.metrics import categorical_accuracy
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras.datasets import cifar10
from sklearn.metrics import accuracy_score

from keras import backend as K

import tensorflow as tf
import numpy as np
import time
import os
import matplotlib.pyplot as plt

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

image_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


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
#_ = compile_and_train(conv_pool_cnn_model, 30)
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

#start2 = time.time()
#_ = compile_and_train(all_cnn_model, 30)
#end2 = time.time()

#print('training time for All - CNN - C: {} s ({} mins.)'.format(end2 - start2, (end2 - start2) / 60.0))

#print('error for All - CNN: ', evaluate_error(all_cnn_model))

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
    x = MaxPooling2D(pool_size = (2, 2))(x)
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

#start3 = time.time()
#_ = compile_and_train(nin_cnn_model, 30)
#end3 = time.time()

#print('training time for NIN - CNN: {} s ({} mins.)'.format(end3 - start3, (end3 - start3) / 60.0))

#print('error for NIN - CNN: ', evaluate_error(nin_cnn_model))


# ensemble of the three models above
conv_pool_cnn_model = conv_pool_cnn(model_input)
all_cnn_model = all_cnn(model_input)
nin_cnn_model = nin_cnn(model_input)

# load best weights for each model (see models folder)
# NOTE: adjust file path names as necessary after training on your own machine
conv_pool_cnn_model.load_weights('weights/conv_pool_cnn.27-0.10.hdf5')
all_cnn_model.load_weights('weights/all_cnn.30-0.07.hdf5')
nin_cnn_model.load_weights('weights/nin_cnn.30-0.86.hdf5')

models = [conv_pool_cnn_model, all_cnn_model, nin_cnn_model]

def ensemble(models, model_input):

    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)

    model = Model(model_input, y, name = 'ensemble')

    return model

ensemble_model = ensemble(models, model_input)

print('error for ensemble of the three models: ', evaluate_error(ensemble_model))

ensemble_preds = ensemble_model.predict(x_test)
ensemble_preds = np.argmax(ensemble_preds, axis = 1)
y_test = y_test.reshape(y_test.shape[0], )

similarity = np.equal(ensemble_preds, y_test)

accuracy = np.sum(similarity) / len(y_test)

print('predictions shape: ', ensemble_preds.shape)
print('target shape: ', y_test.shape)
#equal = K.equal(y_test, np.argmax(ensemble_preds, axis = 1))
#accuracy = tf.reduce_mean(tf.cast(equal, 'float'))
#accuracy = categorical_accuracy(y_test, np.expand_dims(np.argmax(ensemble_preds, axis = 1), axis = 1))

#print('accuracy: ', accuracy)

print('3-model ensemble accuracy: {}%'.format(accuracy * 100))

def targeted_predict(index, predictions, targets):


    print('\n\n---------------------------\n\n')
    print('predicted: {} ({})'.format(image_classes[predictions[index]], predictions[index]))
    print('actual: {} ({})'.format(image_classes[targets[index]], targets[index]))

    plt.imshow(x_test[index])
    plt.show()

# tests
targeted_predict(10, ensemble_preds, y_test)
targeted_predict(233, ensemble_preds, y_test)
targeted_predict(5679, ensemble_preds, y_test)
targeted_predict(4832, ensemble_preds, y_test)
targeted_predict(4911, ensemble_preds, y_test)
targeted_predict(6082, ensemble_preds, y_test)
targeted_predict(9262, ensemble_preds, y_test)
targeted_predict(2072, ensemble_preds, y_test)
targeted_predict(8112, ensemble_preds, y_test)
targeted_predict(3034, ensemble_preds, y_test)