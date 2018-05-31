from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage import measure
import matplotlib.pyplot as plt
import matplotlib.patches as patches

car_image = imread('bmw_license_plate.jpg', as_gray = True)
print(car_image.shape)

# change range from [0,1] to [0, 255]
grayscale_image = car_image * 255
#fig, (ax1, ax2) = plt.subplots(1, 2)
#ax1.imshow(grayscale_image, cmap = 'gray')
threshold_value = threshold_otsu(grayscale_image)
binary_car_image = grayscale_image > threshold_value
#ax2.imshow(binary_car_image, cmap = 'gray')

# connected component analysis to form sections in the image
label_image = measure.label(binary_car_image)
fig, (ax1) = plt.subplots(1)
ax1.imshow(grayscale_image, cmap = 'gray')

# finding max height, width, and min height, width of license plate
# assumes width is between 15 - 40 % of the total image's size
# and height is between 8 - 20 % of the total image's size
plate_dimensions = (0.05 * label_image.shape[0], 0.2 * label_image.shape[0], 0.1 * label_image.shape[1], 0.4 * label_image.shape[1])
min_height, max_height, min_width, max_width = plate_dimensions

plate_objects_coordinates = []
plate_like_objects = []

# list of properties of labeled regions
for region in measure.regionprops(label_image):
    # areas this small are likely not the license plate
    if region.area < 50:
        continue

    # get opposite corners of the bounding box
    minRow, minCol, maxRow, maxCol = region.bbox
    region_height = maxRow - minRow
    region_width = maxCol - minCol

    # check to ensure it satisfies the license plate conditions above
    if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width:
        plate_like_objects.append(binary_car_image[minRow: maxRow, minCol: maxCol])
        plate_objects_coordinates.append((minRow, minCol, maxRow, maxCol))

        rectBorder = patches.Rectangle((minCol, minRow), maxCol - minCol, maxRow - minRow, edgecolor = 'red', linewidth = 2, fill = False)
        ax1.add_patch(rectBorder)




plt.show()