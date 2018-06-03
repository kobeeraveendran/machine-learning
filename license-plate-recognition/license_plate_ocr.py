import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.filters import threshold_otsu
from skimage import util
from skimage import measure
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
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
        plate_like_objects.append(binary_car_image[minRow:maxRow, minCol:maxCol])
        plate_objects_coordinates.append((minRow, minCol, maxRow, maxCol))
        print(region.bbox)

        rectBorder = patches.Rectangle((minCol, minRow), maxCol - minCol, maxRow - minRow, edgecolor = 'red', linewidth = 2, fill = False)
        ax1.add_patch(rectBorder)

# from observing the image with bounding boxes, I determined the license plate was plate_like_objects[4]
# the need for manual observation will be removed later
license_plate = util.invert(plate_like_objects[4])

labelled_plate = measure.label(license_plate)

fig, (ax2) = plt.subplots(1)
ax2.imshow(license_plate, cmap = 'gray')

character_dimensions = (0.35 * license_plate.shape[0], 0.90 * license_plate.shape[0], 0.05 * license_plate.shape[1], 0.90 * license_plate.shape[1])
min_height, max_height, min_width, max_width = character_dimensions

characters = []
counter = 0
column_list = []

for regions in measure.regionprops(labelled_plate):
    y0, x0, y1, x1 = regions.bbox
    region_width = x1 - x0
    region_height = y1 - y0

    if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
        roi = license_plate[y0:y1, x0:x1]

        rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor = 'red', linewidth = 2, fill = False)
        
        ax2.add_patch(rect_border)

        resized_char = resize(roi, (20, 20))
        characters.append(resized_char)

        column_list.append(x0)

# character recognition

chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K','L', 
        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def read_training_data(training_directory):
    image_data = []
    target_data = []

    for char in chars:
        for x in range(10):
            image_path = os.path.join(training_directory, char, char + '_' + str(x) + '.jpg')
            img_details = imread(image_path, as_grey = True)
            binary_image = threshold_otsu(img_details) > img_details

            flat_binary_image = binary_image.reshape(-1)
            image_data.append(flat_binary_image)
            target_data.append(char)

    return (np.array(image_data), np.array(target_data))

def cross_validation(mode, num_folds, train_data, train_label):
    accuracy_result = cross_val_score(mode, train_data, train_label, cv = num_folds)

    print('Cross validation result for ' + str(num_folds) + '-fold: ')

    print(accuracy_result * 100)

current_dir = os.path.dirname(os.path.realpath(__file__))

training_dataset_dir = os.path.join(current_dir, 'train20X20')

image_data, target_data = read_training_data(training_dataset_dir)

svc_model = SVC(kernel = 'linear', probability = True)

cross_validation(svc_model, 4, image_data, target_data)

svc_model.fit(image_data, target_data)

# save the trained model so that re-training is unnecessary
save_directory = os.path.join(current_dir, 'models/svc/')

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

joblib.dump(svc_model, save_directory + '/svc.pkl')

plt.show()