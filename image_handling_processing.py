# using PIL to read images
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('empire.jpg')

# color conversions
# convert the image read above to grayscale
img_grayscale = img.convert('L')

img_grayscale.show()

# convert to another format

# takes all image files in a list of filenames and
# converts them to a specified image file format
import os

filelist = ['empire.jpg', 'bmw_license_plate.jpg']

for file in filelist:
    newfile = os.path.splitext(file)[0] + '.png'
    if file != newfile:
        try:
            Image.open(file).save(newfile)
        except IOError:
            print('Cannot convert ' + file)