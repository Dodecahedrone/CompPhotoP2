from HDR_helperfiles import *

from PIL import Image
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import cv2

#create an array to hold the raw images
raw_images = np.zeros((16, 4016, 6016, 3), dtype=np.float32)

#array to hold exposure data
exposures = np.empty(16)

# z_min and z_max variables
z_min = 0.05
z_max = 0.95


#load each image and exposure info into respective arrays
for i in range(0,16):
    filename = './exposure' + str(i+1) + '.tiff'
    im = readHDR(filename)
    exposures[i] = get_exposure(filename)
    raw_images[i] = np.array(im)/65535
    '''
    print(exposures[i])
    plt.imshow(raw_images[i])
    plt.show()
    '''

# define weighting functions
# photon weight uses gaussian weight but multiplied by exposure
def uniform_weight(x):
    if z_min <= x <= z_max:
        return 1
    else:
        return 0

def tent_weight(x):
    if z_min <= x <= z_max:
        return np.minimum(x, 1-x)
    else:
        return 0

def gaussian_weight(x):
    if z_min <= x <= z_max:
        return np.exp(-4 * ((x - 0.5) ** 2) / 0.25)
    else:
        return 0

# vectorize functions
uniform_weight_vectorized = np.vectorize(uniform_weight)
tent_weight_vectorized = np.vectorize(tent_weight)
gaussian_weight_vectorized = np.vectorize(gaussian_weight)

# create list of functions to use and final HDR images
functions = [uniform_weight_vectorized, tent_weight_vectorized, gaussian_weight_vectorized, uniform_weight_vectorized]
HDR_images = np.zeros((16, 4016, 6016, 3), dtype=np.float32)


#empty color channels
red = raw_images[:,:,:,0]
print("red complete")
green = raw_images[:,:,:,1]
print("blue complete")
blue = raw_images[:,:,:,2]
print("green complete")

channels = [red, green, blue]

# apply 4 weighting functions
for f in range(0,4):
    final_channels = []

    # loop for 3 color channels
    for channel in channels:
        #get the weights of the image
        weighted = functions[f](channel)
        print("weights applied")

        #remove noisy and clipped values by multiplying channel and weights
        multiply =  channel * weighted
        print("multiply complete")

        #divide by the respective exposure time
        for i in range(0, 16):
            multiply[i] /= exposures[i]
        print("exposure complete")

        # for photon weighting, multiply weights by respective exposure time
        if f == 3:
            for i in range(0, 16):
                weighted *= exposures[i]
            print("photon weight applied")

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

        #add channel to final average
        final_channels.append(averaged)

    # combine channels
    final_channels_np = np.array(final_channels)
    final_image = np.stack(final_channels_np, axis=-1)

    # store final HDR image
    HDR_images[f] = final_image

    # display image
    plt.imshow(final_image)
    plt.show()

