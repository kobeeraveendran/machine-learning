import numpy as np
from rnn_utils import softmax, sigmoid, initialize_adam, update_parameters_with_adam

# RNN operations for a single time-step
def rnn_cell_forward(xt, a_prev, parameters):

    Wax = parameters['Wax']
    Waa = parameters['Waa']
    Wya = parameters['Wya']
    ba = parameters['ba']
    by = parameters['by']

    a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)

    yt_pred = softmax(np.dot(Wya, a_next) + by)

    cache = (a_next, a_prev, xt, parameters)

    return a_next, yt_pred, cache

# check rnn ops for single time step
np.random.seed(1)
xt = np.random.randn(3, 10)
a_prev = np.random.randn(5, 10)
Waa = np.random.randn(5, 5)
Wax = np.random.randn(5, 3)
Wya = np.random.randn(2, 5)
ba = np.random.randn(5, 1)
by = np.random.randn(2, 1)
parameters = {'Waa': Waa, 'Wax': Wax, 'Wya': Wya, 'ba': ba, 'by': by}

a_next, yt_pred, cache = rnn_cell_forward(xt, a_prev, parameters)

print('a_next[4] = ', a_next[4])
print('a_next shape: ', a_next.shape)
print('yt_pred[1] = ', yt_pred[1])
print('yt_pred shape: ', yt_pred.shape)


# forward prop (cycle the single-step RNN cells over T time-steps)
def rnn_forward(x, a0, parameters):

    caches = []

    n_x, m, T_x = x.shape
    n_y, n_a = parameters['Wya'].shape

    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))

    a_next = a0

    for t in range(T_x):

        a_next, yt_pred, cache = rnn_cell_forward(x[..., t], a_next, parameters)
        a[..., t] = a_next
        y_pred[..., t] = yt_pred

        caches.append(cache)

    caches = (caches, x)

    return a, yt_pred, caches