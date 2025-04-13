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

# crop values for smaller images during debugging
crop_x = 2000
crop_y = 1333

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

raw_images_cropped = raw_images[:,:crop_y,:crop_x,:]

# define weighting functions
# photon weight uses gaussian weight but multiplied by exposure
def uniform_weight(x):
    if z_min <= x <= z_max:
        return float(1)
    else:
        return float(0)

def tent_weight(x):
    if z_min <= x <= z_max:
        return float(np.minimum(x, 1-x))
    else:
        return float(0)

def gaussian_weight(x):
    if z_min <= x <= z_max:
        return float(np.exp(-4 * ((x - 0.5) ** 2) / 0.25))
    else:
        return float(0)

# vectorize functions
uniform_weight_vectorized = np.vectorize(uniform_weight)
tent_weight_vectorized = np.vectorize(tent_weight)
gaussian_weight_vectorized = np.vectorize(gaussian_weight)

# create list of functions to use and final HDR images
functions = [uniform_weight_vectorized, tent_weight_vectorized, gaussian_weight_vectorized, uniform_weight_vectorized]
#HDR_images = np.zeros((4, 4016, 6016, 3), dtype=np.float32)
HDR_images = np.zeros((4, crop_y, crop_x, 3), dtype=np.float32)


#empty color channels
red = raw_images_cropped[:,:,:,0]
green = raw_images_cropped[:,:,:,1]
blue = raw_images_cropped[:,:,:,2]

channels = [red, green, blue]


# apply 4 weighting functions
for f in range(0,4):
    final_channels = []

    # loop for 3 color channels
    for channel in channels:
        #get the weights of the image
        weighted = functions[f](channel)

        print("weights applied")
        print(type(weighted[0][0][0]))

        if f == 3:
            weighted *= exposures[:, None, None]
            print("photon weight applied")

        #remove noisy and clipped values by multiplying channel and weights
        multiply =  channel * weighted
        print("multiply complete")
        print(type(multiply[0][0][0]))

        #divide by the respective exposure time
        multiply /= exposures[:, None, None]
        print(type(exposures[0]))
        print("exposure complete")

        #sum weights to get denominator
        #summed = reduce(lambda x, y: x + y, weighted)
        denominator = np.sum(weighted, axis=0) + 1e-8
        print("weights summed")
        print(type(denominator[0][0]))

        #sum channel to get numerator
        #channel_summed = reduce(lambda x, y: x + y, multiply)
        numerator = np.sum(multiply, axis=0)
        print(type(numerator[0][0]))
        print("channel summed")

        #compute average
        averaged = numerator / denominator
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


