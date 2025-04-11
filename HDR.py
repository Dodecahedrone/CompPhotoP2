from HDR_helperfiles import *

from PIL import Image
import numpy as np
import cv2

#create an array to hold the raw images
raw_images = np.empty((16, 4016, 6016, 3))

#array to hold exposure data
exposures = np.empty(16)

z_min = 0.05
z_max = 0.95

#load each image into the array
for i in range(5,10):
    filename = './exposure' + str(i+1) + '.tiff'
    im = Image.open(filename)
    exposures[i] = get_exposure(filename)
    raw_images[i] = (np.array(im)/255)


def uniform_weight(x):
    if z_min <= x <= z_max:
        return 1
    else:
        return 0

#empty final image
final_image = np.empty((4016, 6016, 3))

# merge HDR
for i in range(0,4016): #for each pixel
    for j in range(0,6016): #for each pixel
        for k in range(5,10): #for each image
            sum_numerator = 0
            sum_denominator = 0
            for c in range (0,3): #for each color channel
                pixel = raw_images[k][i][j][c]
                sum_numerator += (pixel * uniform_weight(pixel)) / exposures[k]
                sum_denominator += uniform_weight(pixel)
                final_image[i][j][c] = sum_numerator / sum_denominator



print(raw_images[15][0][0])
print(get_exposure('./exposure16.tiff'))




'''
Each pixel is the sum of the weighting function of that pixel * that pixel * division divided by sum of weighting function
'''
