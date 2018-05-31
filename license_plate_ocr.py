from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage import measure
import matplotlib.pyplot as plt
import matplotlib.patches as patches

car_image = imread('bmw_license_plate.jpg', as_gray = True)
print(car_image.shape)

# change range from [0,1] to [0, 255]
grayscale_image = car_image * 255
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(grayscale_image, cmap = 'gray')
threshold_value = threshold_otsu(grayscale_image)
binary_car_image = grayscale_image > threshold_value
ax2.imshow(binary_car_image, cmap = 'gray')

# connected component analysis to form sections in the image
label_image = measure.label(binary_car_image)
fig, (ax1) = plt.subplots(1)
ax1.imshow(grayscale_image, cmap = 'gray')
plt.show()