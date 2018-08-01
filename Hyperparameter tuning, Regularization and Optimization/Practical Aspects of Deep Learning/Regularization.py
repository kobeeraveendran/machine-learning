import numpy as np
import matplotlib.pyplot as plt
from reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec
from reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters
import sklean
import sklearn.datasets
import scipy.io
from testCases import *

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_X, train_Y, test_X, test_Y = load_2D_dataset()

def model(X, Y, learning_rate = 0.3, num_iterations = 30000, print_cost = True, lambd = 0, keep_prob = 1):
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0], 20, 3, 1]

    parameters = initialize_parameters(layers_dims)

    for i in range(num_iterations):
        if keep_prob == 1:
            a3, cache = forward_propagation(X, parameters)
        
        elif keep_prob < 1:
            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)

        if lambd == 0:
            cost = compute_cost(a3, Y)
        else:
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)

        if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(X, Y, cache)
        
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)

        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)

        if print_cost and i % 10000 == 0:
            print('Cost after iteration {}: {}'.format(i, cost))
            costs.append(cost)

        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (in ten thousands)')
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()

        return parameters

def compute_cost_with_regularization(A3, Y, parameters, lambd):
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    cross_entropy_cost = compute_cost(A3, Y)

    L2_regularization_cost = (1.0 / m) * (lambd / 2) * np.sum([np.sum(np.square(i)) for i in [W1, W2, W3]])

    cost = cross_entropy_cost + L2_regularization_cost

    return cost


parameters = model(train_X, train_Y)
print("Training set: ")
predictions_train = predict(train_X, train_Y, parameters)
print("Testing set: ")
predictions_test = predict(test_X, test_Y, parameters)

plt.title("Model without regularization")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.40])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

##########################
# WITH L2 REGULARIZATION #
##########################

A3, Y_assess, parameters = compute_cost_with_regularization_test_case()
print("Cost = " + str(compute_cost_with_regularization(A3, Y_assess, parameters, lambd = 0.1)))

