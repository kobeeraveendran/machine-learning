import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
from init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# load training and testing sets
train_X, train_Y, test_X, test_Y = load_dataset()

def model(X, Y, learning_rate = 0.01, num_iterations = 15000, print_cost = True, initialization = 'he'):

    grads = {}
    costs = []

    m = X.shape[1]
    layers_dims = [X.shape[0], 10, 5, 1]

    if initialization == 'zeros':
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == 'random':
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == 'he':
        parameters = initialize_parameters_he(layers_dims)

    for i in range(0, num_iterations):
        a3, cache = forward_propagation(X, parameters)

        cost = compute_loss(a3, Y)

        grads = backward_propagation(X, Y, cache)

        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print('Cost after iteration {}: {}'.format(i, cost))
            costs.append(cost)

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (in hundreds)')
    plt.title('Learning rate = {}'.format(learning_rate))
    plt.show()

    return parameters


def initialize_parameters_zeros(layers_dims):
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.zeros(shape = (layers_dims[l], layers_dims[l - 1]))
        parameters['b' + str(l)] = np.zeros(shape = (layers_dims[l], 1))

    return parameters

# ZEROS INITIALIZATION

parameters = initialize_parameters_zeros([3, 2, 1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

# training and testing predictions
parameters = model(train_X, train_Y, initialization = 'zeros')
print("Training set: ")
predictions_train = predict(train_X, train_Y, parameters)
print("Testing set: ")
predictions_test = predict(test_X, test_Y, parameters)

# debug - both are all 0s
print("predictions_train = " + str(predictions_train))
print("predictions_test = " + str(predictions_test))

# neurons all predict 0 (they each learn the same weights)
# avoid by initializing weights randomly (biases may be initialized to zero though)
plt.title('Model with zeros initialization')
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)