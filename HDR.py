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
crop_x = 6016
crop_y = 4016

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

        # create a mask for noisy and clipped values
        mask = np.ones(channel.shape)
        mask[channel > z_max] = 0.0
        mask[channel < z_min] = 0.0

        weighted = np.empty(channel.shape)

        # apply the appropriate weighting function
        if f == 0:
            weighted = mask
        elif f == 1:
            tent = (-1 * np.abs(channel - 0.5)) + 0.5
            weighted = mask * tent
        elif f == 2:
            gaussian = np.exp(-4 * ((channel - 0.5) ** 2) / 0.25)
            weighted = mask * gaussian
        elif f == 3:
            weighted = mask
            weighted *= exposures[:, None, None]

        print("weights applied")

        #remove noisy and clipped values by multiplying channel and weights
        multiply =  channel * weighted
        print("multiply complete")

        #divide by the respective exposure time
        multiply /= exposures[:, None, None]
        print("exposure complete")

        #sum weights to get denominator
        #summed = reduce(lambda x, y: x + y, weighted)
        denominator = np.sum(weighted, axis=0) + 1e-8
        print("weights summed")

        #sum channel to get numerator
        #channel_summed = reduce(lambda x, y: x + y, multiply)
        numerator = np.sum(multiply, axis=0)
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


