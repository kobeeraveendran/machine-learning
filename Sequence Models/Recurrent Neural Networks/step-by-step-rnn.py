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

print('\nSingle-step RNN cell check:')

np.testing.assert_array_almost_equal(a_next[4], [0.59584544, 0.18141802, 0.61311866, 0.99808218, 0.85016201, 0.99980978, -0.18887155, 0.99815551, 0.6531151, 0.82872037], err_msg = 'Failed.')
print('Passed.')

print('Passed.') if a_next.shape == (5, 10) else print('Failed.')

np.testing.assert_array_almost_equal(yt_pred[1], [0.9888161, 0.01682021, 0.21140899, 0.36817467, 0.98988387, 0.88945212, 0.36920224, 0.9966312, 0.9982559, 0.17746526], err_msg = 'Failed.')
print('Passed.')

print('Passed.') if yt_pred.shape == (2, 10) else print('Failed.')


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

    return a, y_pred, caches

# check rnn forwardprop
np.random.seed(1)
x = np.random.randn(3, 10, 4)
a0 = np.random.randn(5, 10)
Waa = np.random.randn(5, 5)
Wax = np.random.randn(5, 3)
Wya = np.random.randn(2, 5)
ba = np.random.randn(5, 1)
by = np.random.randn(2, 1)
parameters = {'Waa': Waa, 'Wax': Wax, 'Wya': Wya, 'ba': ba, 'by': by}

a, y_pred, caches = rnn_forward(x, a0, parameters)

print('\n\nRNN Forwardprop check:')

np.testing.assert_array_almost_equal(a[4][1], [-0.99999375, 0.77911235, -0.99861469, -0.99833267], err_msg = 'Failed.')
print('Passed.')

print('Passed.') if a.shape == (5, 10, 4) else print('Failed.')

np.testing.assert_array_almost_equal(y_pred[1][3], [0.79560373, 0.86224861, 0.11118257, 0.81515947], err_msg = 'Failed.')
print('Passed.')

print('Passed.') if y_pred.shape == (2, 10, 4) else print('Failed.')

np.testing.assert_array_almost_equal(caches[1][1][3], [-1.1425182, -0.34934272, -0.20889423, 0.58662319], err_msg = 'Failed.')
print('Passed.')

print('Passed.') if len(caches) == 2 else print('Failed.')