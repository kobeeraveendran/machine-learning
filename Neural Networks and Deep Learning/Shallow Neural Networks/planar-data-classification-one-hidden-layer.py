import numpy as np 
import matplotlib.pyplot as plt 
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

# NOTE: FOR DISPLAYING MATPLOTLIB VISUALS, RUN IN JUPYTER NOTEBOOK,
# UNCOMMENT THE LINE BELOW, AND UNCOMMENT ALL MATPLOTLIB FUNCTIONS BELOW
#%matplotlib inline

# for consistent reuslts
np.random.seed(1)

# load in flower 2-class dataset
X, Y = load_planar_dataset()

# visualize data
#plt.scatter(X[0, :], X[1, :], c = Y, s = 40, cmap = plt.cm.Spectral)

# for getting hapes of numpy arrays
shape_X = X.shape
shape_Y = Y.shape 
m = Y.shape[1]

# neural network model helper functions below

# define neural network structure by computing and returning three variables:
# n_x (size of input layer), n_h (size of hidden layer), and n_y (size of output layer)
def layer_sizes(X, Y):
    '''
    X - input dataset, with shape (input size, number of examples)
    Y - labels, with shape (output size, number of examples)
    '''

    n_x = X.shape[0]
    n_h = 4 # this value was hard coded to use 4 hidden layers
    n_y = Y.shape[0]

    return n_x, n_h, n_y

# initialize parameters
# weights will be initialized to random numbers, and biases will be 0's
def initialize_parameters(n_x, n_h, n_y):
    '''
    returns:
    parameters - python dictionary of the following parameters:
        W1 - matrix of weights, with shape (n_h, n_x)
        b1 - vector of biases, with shape (n_h, 1)
        W2 - matrix of weights, with shape (n_y, n_h)
        b2 - vector of biases, with shape (n_y, 1)
    '''

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    return parameters

# forward propagation function
# see the mathematical representation of the process below for details/intuition
# z[1](i) = W[1]x(i) + b[1]
# a[1](i) = tanh(z[1](i))
# z[2](i) = W[2]a[1](i) + b[2]
# y(i) = a[2](i) = sigmoid(z[2](i))

def forward_propagation(X, parameters):
    '''
    returns:
    A2 - sigmoid output of the second activation function
    cache - dictionary containing the four computations above (Z1, A1, Z2, A2 (as y(i)))
    '''

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    # forward propagation steps as defined by the above equations
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

    return A2, cache

# cost function implementation using the cost equation:
# -1/m * sum(y(i)log(a[2](i)) + (1 - y(i))log(1 - a[2](i)))
# look up 'cross-entropy cost function' for easier-to-understand version
def compute_cost(A2, Y, parameters):
    '''
    returns:
    cost - cross entropy cost given by the above equation
    '''

    # find the value of m in the above formula (number of examples)
    m = Y.shape[1]

    # compute cross entropy cost
    # start with inner part of equation, then sum
    logprobs = np.multiply(Y, np.log(A2)) + np.multiply(1 - Y, np.log(1 - A2))
    cost = -1 * np.sum(logprobs) / m

    # make sure cost is the size we expect by removing single-dimensional entries
    # (i.e. [[17]] -> 17)
    cost = np.squeeze(cost)

    return cost
    
# backward propagation function
# uses the 6 equations on RHS of summary-gradient-descent.png
# can also derive by taking derivatives
# note: g[1]() is the activation function tanh, so if a = g[1](z), then g[1]'(z) = 1 - a^2
# thus, g[1]'(Z[1]) = 1 - A1^2
def backward_propagation(parameters, cache, X, Y):
    '''
    returns:
    grads - dictionary containing the calculated gradients w.r.t. different parameters
    '''

    # find value of m in the equations
    m = X.shape[1]

    # grab any parameters that we'll need in the equations
    W2 = parameters['W2']
    A1 = cache['A1']
    A2 = cache['A2']

    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis = 1, keepdims = True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return grads

# function to update parameters after backpropagation
# uses the gradient descent update rule: theta = theta - (alpha)(dJ/dtheta)
# where alpha = learning rate, dJ/dtheta = respective gradients, and theta = respective parameters

def update_parameters(parameters, grads, learning_rate = 1.2):
    '''
    returns:
    parameters - dictionary with the updated parameters
    '''

    # retreive parameters for udpating
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    # retreive calculated gradients
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    # apply the above gradient descent rules and update parameter values
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    # update the parameters in dictionary
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    return parameters

# finally, build up all the previous helper functions and apply them in a NN model
def nn_model(X, Y, n_h, num_iterations = 10000, print_cost = False):
    '''
    recap of Args:
    X - dataset, with shape (2, num examples)
    Y - labels, with shape (1, num examples)
    n_h - size (number of nodes) of the hidden layer
    num_iterations - number of iterations in the gradient descent loop
    print_cost - boolean to control cost printing every 1000 iterations, False by default

    returns:
    parameters - parameters learned by the model, to be used later in prediction
    '''

    # for consistency
    np.random.seed(3)
    
    # determine structure using the layer_sizes() above
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    # initialize parameters
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters['W1']
    b1 = parameters['b2']
    W2 = parameters['W2']
    b2 = parameters['b2']

    # perform gradient descent loop
    for i in range(0, num_iterations):
        # forward propagation stage
        A2, cache = forward_propagation(X, parameters)

        # cost function computation
        cost = compute_cost(A2, Y, parameters)

        # backward propagation stage
        grads = backward_propagation(parameters, cache, X, Y)

        # update parameters using grads from backprop
        parameters = update_parameters(parameters, grads)

        # if print_cost is True
        if print_cost and i % 1000 == 0:
            print('Cost after iteration ' + str(i) + ': ' + str(cost))
        
    return parameters

# prediction function
# if activation > 0.5 -> 1, else -> 0

def predict(parameters, X):
    '''
    returns:
    predictions - vector of predictions from the model
    '''

    # compute probabilities using forward prop, classifies based on value of activation
    A2, cache = forward_propagation(X, parameters)
    # apply round to A2, which produces vector with 1 if A2[i] > 0.5 or 0 otherwise
    predictions = np.round(A2)

    return predictions

# test run of the model with just one hidden layer
parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost = True)

# plot decision boundary
#plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
#plt.title("Decision Boundary for Hidden Layer Size " + str(4))

# print accuracy
predictions = predict(parameters, X)
print('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')

# testing on different numbers of hidden layers
plt.figure(figsize = (16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]

for i, n_h in enumerate(hidden_layer_sizes):
    #plt.subplot(5, 2, i + 1)
    #plt.title('Hidden layer of size ' + str(n_h))
    parameters = nn_model(X, Y, n_h, num_iterations = 5000)
    #plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T))/float(Y.size) * 100)
    print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
