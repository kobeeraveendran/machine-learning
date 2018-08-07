import numpy as np
import matplotlib.pyplot as plt
import h5py

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

# zero-padding
def zero_pad(X, pad):
    
    X_pad = np.pad(X, pad_width = ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode = 'constant', constant_values = 0)

    return X_pad

# check padding
np.random.seed(1)
x = np.random.randn(4, 3, 3, 2)
x_pad = zero_pad(x, 2)
print('x.shape: ' + str(x.shape))
print('x_pad.shape: ' + str(x_pad.shape))
print('x[1, 1]: ' + str(x[1, 1]))
print('x_pad[1, 1]: ' + str(x_pad[1, 1]))

fig, axarr = plt.subplots(nrows = 1, ncols = 2)
axarr[0].set_title('x')
axarr[0].imshow(x[0, :, :, 0])
axarr[1].set_title('x_pad')
axarr[1].imshow(x_pad[0, :, :, 0])
plt.show()

def conv_single_step(a_slice_prev, W, b):
    s = np.multiply(a_slice_prev, W)
    Z = np.sum(s)
    Z += b

    return Z

np.random.seed(1)
a_slice_prev = np.random.randn(4, 4, 3)
W = np.random.randn(4, 4, 3)
b = np.random.randn(1, 1, 1)

Z = conv_single_step(a_slice_prev, W, b)
print('Z: ' + str(Z))

def conv_forward(A_prev, W, b, hyperparameters):

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape[0], A_prev.shape[1], A_prev.shape[2], A_prev.shape[3]

    (f, f, n_C_prev, n_C) = W.shape[0], W.shape[1], W.shape[2], W.shape[3]

    stride = hyperparameters['stride']
    pad = hyperparameters['pad']

    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1

    Z = np.zeros(shape = (m, n_H, n_W, n_C))

    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):

                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    a_slice_prev = a_prev_pad[vert_start: vert_end, horiz_start: horiz_end, :]

                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:, :, :, c], b[:, :, :, c])

    assert(Z.shape == (m, n_H, n_W, n_C))

    cache = (A_prev, W, b, hyperparameters)

    return Z, cache

# check implementation of conv_forward
np.random.seed(1)
A_prev = np.random.randn(10, 4, 4, 3)
W = np.random.randn(2, 2, 3, 8)
b = np.random.randn(1, 1, 1, 8)

hyperparameters = {'pad': 2, 'stride': 2}

Z, cache_conv = conv_forward(A_prev, W, b, hyperparameters)

print("Z's mean: " + str(np.mean(Z)))
print("Z[3, 2, 1]: " + str(Z[3, 2, 1]))
print("cache_conv[0][1][2][3]: " + str(cache_conv[0][1][2][3]))


def pool_forward(A_prev, hyperparameters, mode = 'max'):

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    f = hyperparameters['f']
    stride = hyperparameters['stride']

    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    A = np.zeros(shape = (m, n_H, n_W, n_C))

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    a_prev_slice = A_prev[i, vert_start: vert_end, horiz_start: horiz_end, c]

                    if mode == 'max':
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == 'average':
                        A[i, h, w, c] = np.mean(a_prev_slice)

    cache = (A_prev, hyperparameters)

    assert(A.shape == (m, n_H, n_W, n_C))

    return A, cache

# check pooling forward
np.random.seed(1)
A_prev = np.random.randn(2, 4, 4, 3)
hyperparameters = {'stride': 2, 'f': 3}
A, cache = pool_forward(A_prev, hyperparameters, mode = 'max')
print('mode = max')
print('A = ' + str(A))
print('\n')

A, cache = pool_forward(A_prev,  hyperparameters, mode = 'average')
print('mode = average')
print('A = ' + str(A))
print('\n')


# backpropagation on convolutional layers
def conv_backward(dZ, cache):

    (A_prev, W, b, hyperparameters) = cache

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    (f, f, n_C_prev, n_C) = W.shape

    stride = hyperparameters['stride']
    pad = hyperparameters['pad']

    (m, n_H, n_W, n_C) = dZ.shape

    dA_prev = np.zeros(shape = A_prev.shape)
    dW = np.zeros(shape = (W.shape))
    db = np.zeros(shape = (1, 1, 1, n_C))

    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):

        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    a_slice = a_prev_pad[vert_start: vert_end, horiz_start: horiz_end, :]

                    da_prev_pad[vert_start: vert_end, horiz_start: horiz_end, :] += W[..., c] * dZ[i, h, w, c]
                    dW[..., c] += a_slice * dZ[i, h, w, c]
                    db[..., c] += dZ[i, h, w, c]

        dA_prev[i, ...] = da_prev_pad[pad: -pad, pad: -pad, :]

    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

    return dA_prev, dW, db

# check conv_backward
np.random.seed(1)
dA, dW, db = conv_backward(Z, cache_conv)
print('dA mean = ' + str(np.mean(dA)))
print('dW mean = ' + str(np.mean(dW)))
print('db mean = ' + str(np.mean(db)))

# backpropagation on pooling layers
# (to propagate gradients to previous layers)

# mask for max pooling
def create_mask_from_window(x):

    mask = (x == np.max(x))

    return mask

# check masking function
np.random.seed(1)
x = np.random.randn(2, 3)
mask = create_mask_from_window(x)
print('x = ' + str(x))
print('mask = ' + str(mask))

# backprop on average pooling layer
def distribute_value(dz, shape):
    n_H, n_W = shape

    average = dz / (n_H * n_W)

    a =  np.ones(shape) * average
    
    return a

a = distribute_value(2, (2, 2))
print('distributed value = ' + str(a))

# tying the pooling methods together
def pool_backward(dA, cache, mode = 'max'):

    A_prev, hyperparameters = cache

    stride = hyperparameters['stride']
    f = hyperparameters['f']

    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape

    dA_prev = np.zeros(shape = A_prev.shape)

    for i in range(m):
        a_prev = A_prev[i]

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):

                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    if mode == 'max':
                        a_prev_slice = a_prev[vert_start: vert_end, horiz_start: horiz_end, c]
                        mask = create_mask_from_window(a_prev_slice)

                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask * dA[i, h, w, c]

                    elif mode == 'average':
                        da = dA[i, h, w, c]
                        shape = (f, f)

                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape)

    assert(dA_prev.shape == A_prev.shape)

    return dA_prev

