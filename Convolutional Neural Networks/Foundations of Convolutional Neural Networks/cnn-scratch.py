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

