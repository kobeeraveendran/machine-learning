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


# LSTM single-step cell

def lstm_cell_forward(xt, a_prev, c_prev, parameters):

    Wf = parameters['Wf']
    bf = parameters['bf']
    Wi = parameters['Wi']
    bi = parameters['bi']
    Wc = parameters['Wc']
    bc = parameters['bc']
    Wo = parameters['Wo']
    bo = parameters['bo']
    Wy = parameters['Wy']
    by = parameters['by']

    #n_x, m = xt.shape
    #n_y, n_a = Wy.shape

    concat = np.vstack((a_prev, xt))

    ft = sigmoid(np.dot(Wf, concat) + bf)
    it = sigmoid(np.dot(Wi, concat) + bi)
    cct = np.tanh(np.dot(Wc, concat) + bc)
    c_next = ft * c_prev + it * cct
    ot = sigmoid(np.dot(Wo, concat) + bo)
    a_next = ot * np.tanh(c_next)

    yt_pred = softmax(np.dot(Wy, a_next) + by)

    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache

np.random.seed(1)
xt = np.random.randn(3, 10)
a_prev = np.random.randn(5, 10)
c_prev = np.random.randn(5, 10)
Wf = np.random.randn(5, 5 + 3)
bf = np.random.randn(5, 1)
Wi = np.random.randn(5, 5 + 3)
bi = np.random.randn(5, 1)
Wo = np.random.randn(5, 5 + 3)
bo = np.random.randn(5, 1)
Wc = np.random.randn(5, 5 + 3)
bc = np.random.randn(5, 1)
Wy = np.random.randn(2, 5)
by = np.random.randn(2, 1)

parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)

print('\n\nLSTM single cell forward prop check:')

np.testing.assert_array_almost_equal(a_next[4], [-0.66408471, 0.0036921, 0.02088357, 0.22834167, -0.85575339, 0.00138482, 0.76566531, 0.34631421, -0.00215674, 0.43827275], err_msg = 'Failed.')
print('Passed.')

print('Passed.') if a_next.shape == (5, 10) else print('Failed.')

np.testing.assert_array_almost_equal(c_next[2], [0.63267805, 1.00570849, 0.35504474, 0.20690913, -1.64566718, 0.11832942, 0.76449811, -0.0981561, -0.74348425, -0.26810932], err_msg = 'Failed.')
print('Passed.')

print('Passed.') if c_next.shape == (5, 10) else print('Failed.')

np.testing.assert_array_almost_equal(yt[1], [0.79913913, 0.15986619, 0.22412122, 0.15606108, 0.97057211, 0.31146381, 0.00943007, 0.12666353, 0.39380172, 0.07828381], err_msg = 'Failed.')
print('Passed.')

np.testing.assert_array_almost_equal(cache[1][3], [-0.16263996, 1.03729328, 0.72938082, -0.54101719, 0.02752074, -0.30821874, 0.07651101, -1.03752894, 1.41219977, -0.37647422], err_msg = 'Failed.')
print('Passed.')

print('Passed.') if len(cache) == 10 else print('Failed.')


# LSTM over T_x time steps
def lstm_forward(x, a0, parameters):

    caches = []

    n_x, m, T_x = x.shape
    n_y, n_a = parameters['Wy'].shape

    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))

    a_next = a0
    c_next = np.zeros(a0.shape)

    for t in range(T_x):
        a_next, c_next, yt, cache = lstm_cell_forward(x[..., t], a_next, c_next, parameters)

        a[..., t] = a_next
        c[..., t] = c_next
        y[..., t] = yt

        caches.append(cache)

    caches = (caches, x)

    return a, y, c, caches


# check LSTM forward prop
np.random.seed(1)
x = np.random.randn(3, 10, 7)
a0 = np.random.randn(5, 10)
Wf = np.random.randn(5, 5 + 3)
bf = np.random.randn(5, 1)
Wi = np.random.randn(5, 5 + 3)
bi = np.random.randn(5, 1)
Wo = np.random.randn(5, 5 + 3)
bo = np.random.randn(5, 1)
Wc = np.random.randn(5, 5 + 3)
bc = np.random.randn(5, 1)
Wy = np.random.randn(2, 5)
by = np.random.randn(2, 1)

parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

a, y, c, caches = lstm_forward(x, a0, parameters)

print('\n\nLSTM forward propagation multi-step check:')

np.testing.assert_almost_equal(a[4][3][6], 0.172117767533, err_msg = 'Failed.')
print('Passed.')

print('Passed.') if a.shape == (5, 10, 7) else print('Failed.')

np.testing.assert_almost_equal(y[1][4][3], 0.95087346185, err_msg = 'Failed.')
print('Passed.')

print('Passed.') if y.shape == (2, 10, 7) else print('Failed.')

np.testing.assert_array_almost_equal(caches[1][1][1], [ 0.82797464, 0.23009474, 0.76201118, -0.22232814, -0.20075807, 0.18656139, 0.41005165], err_msg = 'Failed.')
print('Passed.')

np.testing.assert_almost_equal(c[1][2][1], -0.855544916718, err_msg = 'Failed.')
print('Passed.')

print('Passed.') if len(caches) == 2 else print('Failed.')