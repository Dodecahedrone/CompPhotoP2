from HDR_helperfiles import *

from PIL import Image
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import cv2

#create an array to hold the raw images
raw_images = np.empty((16, 4016, 6016, 3))

#array to hold exposure data
exposures = np.empty(16)

z_min = 0.05
z_max = 0.95

#load each image into the array
for i in range(0,16):
    filename = './exposure' + str(i+1) + '.tiff'
    im = Image.open(filename)
    exposures[i] = get_exposure(filename)
    raw_images[i] = np.array(im)/255
    print(exposures[i])
    plt.imshow(raw_images[i])
    plt.show()



'''
def uniform_weight(x):
    if z_min <= x <= z_max:
        return 1
    else:
        return 0

#empty final image
red = raw_images[:,:,:,0]
print("red complete")
green = raw_images[:,:,:,1]
print("blue complete")
blue = raw_images[:,:,:,2]
print("green complete")

channels = [red, green, blue]
final_channels = []

#16 images

uniform_weight_vectorized = np.vectorize(uniform_weight)

for channel in channels:
    #get the weights of the image
    weighted = uniform_weight_vectorized(channel)
    print("weights applied")

    #remove noisy and clipped values by multiplying channel and weights
    multiply =  channel * weighted
    print("multiply complete")

    #divide by the respective exposure time
    for i in range(0, 16):
        multiply[i] /= exposures[i]
    print("exposure complete")

    #sum weights to get denominator
    summed = reduce(lambda x, y: x + y, weighted)
    print("weights summed")

    #sum channel to get numerator
    channel_summed = reduce(lambda x, y: x + y, multiply)
    print("channel summed")

    #compute average
    averaged = channel_summed / summed
    averaged = np.nan_to_num(averaged, nan=0)
    print("average computed for channel")

    final_channels.append(averaged)

final_channels_np = np.array(final_channels)
final_image = np.stack(final_channels_np, axis=-1)

plt.imshow(final_image)
plt.show()
'''





'''
uniform_weight_vectorized = np.vectorize(uniform_weight)

uniform_weight_vectorized(raw_images)
'''

'''
# merge HDR
for i in range(0,4016): #for each pixel
    for j in range(0,6016): #for each pixel
        for k in range(0,16): #for each image
            sum_numerator = 0
            sum_denominator = 0
            for c in range (0,3): #for each color channel
                pixel = raw_images[k][i][j][c]
                sum_numerator += (pixel * uniform_weight(pixel)) / exposures[k]
                sum_denominator += uniform_weight(pixel)
                final_image[i][j][c] = sum_numerator / sum_denominator
'''





'''
Each pixel is the sum of the weighting function of that pixel * that pixel * division divided by sum of weighting function
'''
