import numpy as np 
import matplotlib.pyplot as plt 
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# load dataset of cats and non-cats
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# determine size of traininging and test sets, determine number of pixels in image
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

# flatten pixel matrices
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T 
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T 

# standardize dataset
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

# activation function (uses the sigmoid function)
def sigmoid(x):

	sig = 1 / (1 + np.exp(-1 * x))

	return sig

# initializes W matrix and b with zeros
def initialize_with_zeros(dim):

	w = np.zeros((dim, 1))
	b = 0

	# make sure dimensions are correct using assert() [good for debugging later],
	# and that b is either a float or int
	assert(w.shape == (dim, 1))
	assert(isinstance(b, float) or isinstance(b, int))

	return w, b

# handles both forward propagation and backpropagation steps
def propagate(w, b, X, Y):
	
	m = X.shape[1]


	# Forward propagation step
	# apply activation function
	A = sigmoid(np.dot(w.T, X) + b)
	# calculate cost of this iteration
	cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A), axis = 1)

	# Backpropagation step
	# calculate respective derivatives
	dw = (1 / m) * np.dot(X, (A - Y).T)
	db = (1 / m) * np.sum(A - Y, axis = 1)

	# sanity checks for later debugging
	assert(dw.shape == w.shape)
	assert(db.dtype == float)
	cost = np.squeeze(cost)
	assert(cost.shape == ())

	grads = {"dw": dw, "db": db}

	return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):

	costs = []

	for i in range(num_iterations):
		# find cost and gradient for this iteration
		grads, cost = propagate(w, b, X, Y)

		# retrieve derivatives from backprop
		dw = grads["dw"]
		db = grads["db"]

		# update values of w and b in backwards pass
		w = w - learning_rate * dw
		b = b - learning_rate * db

		# record costs
		if i % 100 == 0:
			costs.append(cost)

		# print cost every 50 training iterations
		if print_cost and i % 50 == 0:
			print('Cost after iteration', i, ':', cost)

		params = {"w": w, "b": b}

		grads = {"dw": dw, "db": db}

		return params, grads, costs

# predicts whether object is a cat using probabilities
def predict(w, b, X):
	m = X.shape[1]
	Y_prediction = np.zeros((1, m))
	w = w.reshape(X.shape[0], 1)

	# compute vector of probabilities of being a cat
	A = sigmoid(np.dot(w.T, X) + b)

	for i in range(A.shape[1]):
		# convert probabilities into definitions
		if A[0,i] > 0.5:
			Y_prediction[0,i] = 1
		else:
			Y_prediction[0,i] = 0

	# dimensions sanity check
	assert(Y_prediction.shape == (1, m))

	return Y_prediction

# putting everything together in a model
def model (X_train, Y_train, X_test, Y_test, num_iterations = 2500, learning_rate = 0.1, print_cost = False):

	#initialize parameters
	w, b = initialize_with_zeros(X_train.shape[0])

	# perform gradient descent
	parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost = False)

	# get parameters
	w = parameters['w']
	b = parameters['b']

	# predict on testing and training sets
	Y_prediction_test = predict(w, b, X_test)
	Y_prediction_train = predict(w, b, X_train)

	d = {'costs': costs, 'Y_prediction_test': Y_prediction_test, 
		'Y_prediction_train': Y_prediction_train, 'w': w, 'b': b, 'learning_rate': learning_rate,
		'num_iterations': num_iterations}

	return d

# finally, run the model
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2500, learning_rate = 0.005)