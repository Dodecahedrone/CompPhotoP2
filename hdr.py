
""" This is a module for hdr imaging homework (15-463/663/862, Computational Photography, Fall 2020, CMU).

Modified by cmetzler on 9/13/2022

You can import necessary functions into your code as follows:
from HDR_helperfiles import *

Depends on OpenCV to read/write HDR files"""

import numpy as np
import cv2
import exifread
from PIL import Image

from functools import reduce
import matplotlib.pyplot as plt



def writeHDR(name, data):
    #flip from rgb to bgr for cv2
    cv2.imwrite(name, 255*data[:, :, ::-1].astype(np.float32))
        
def readHDR(name):
    raw_in = cv2.imread(name, flags=cv2.IMREAD_UNCHANGED)
    #flip from bgr to rgb
    return raw_in[:, :, ::-1]

def get_exposure(filename):
    f=open(filename,'rb')
    exposure=exifread.process_file(f)['EXIF ExposureTime'].values[0].decimal()
    return exposure



#part 2

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
crop = True

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
if(crop):
    HDR_images = np.zeros((4, crop_y, crop_x, 3), dtype=np.float32)
else:
    HDR_images = np.zeros((4, 4016, 6016, 3), dtype=np.float32)



#empty color channels
if(crop):
    red = raw_images_cropped[:,:,:,0]
    green = raw_images_cropped[:,:,:,1]
    blue = raw_images_cropped[:,:,:,2]
else:
    red = raw_images[:,:,:,0]
    green = raw_images[:,:,:,1]
    blue = raw_images[:,:,:,2]


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


 # parts 3 and 4

def reinhard_tonemap(hdr, K=0.15, B=0.95, eps=1e-6):
    
    luminance = np.mean(hdr, axis=2) + eps
    log_lum = np.log(luminance)
    avg_log_lum = np.exp(np.mean(log_lum))
    
    scaled_hdr = (K / avg_log_lum) * hdr
    max_val = np.max(scaled_hdr)
    I_white = B * max_val
    
    numerator = scaled_hdr * (1.0 + (scaled_hdr / (I_white**2 + eps)))
    denominator = 1.0 + scaled_hdr
    tone_mapped = numerator / (denominator + eps)
    
    
    tone_mapped = np.clip(tone_mapped, 0.0, 1.0)
    return tone_mapped

def gamma_srgb(img_linear):
    
    mask = (img_linear <= 0.0031308)
    low  = 12.92 * img_linear
    high = 1.055 * (img_linear ** (1.0 / 2.5)) - 0.055
    return np.where(mask, low, high)




K_values = np.linspace(0.15, 2.0, 5)  # create arrays of k and b to iterate over
B_values = np.linspace(0.95, 5.0, 5) 

# iterate k and b over all weights
for idx in range(4):
    hdr_img = HDR_images[idx]  
    
    # display and save both results with and without gamma correction
    for K_val in K_values:
        for B_val in B_values:
            
            
            tm_img = reinhard_tonemap(hdr_img, K=K_val, B=B_val)
            
            
            tm_img_gamma = gamma_srgb(tm_img)
            tm_img_gamma = np.clip(tm_img_gamma, 0.0, 1.0)
            
            
            plt.figure()
            plt.title(f"Weight {idx}, Tone-Mapped Only (K={K_val:.2f}, B={B_val:.2f})")
            plt.imshow(tm_img)
            plt.show()
            
            
            plt.figure()
            plt.title(f"Weight {idx}, Tone-Mapped + Gamma (K={K_val:.2f}, B={B_val:.2f})")
            plt.imshow(tm_img_gamma)
            plt.show()
            
            
            out_name_nogamma = f"hdr_weight{idx}_K{K_val:.2f}_B{B_val:.2f}_nogamma.png"
            writeHDR(out_name_nogamma, tm_img)
            
            
            out_name_gamma = f"hdr_weight{idx}_K{K_val:.2f}_B{B_val:.2f}_gamma.png"
            writeHDR(out_name_gamma, tm_img_gamma)

            print("Saved:", out_name_nogamma, "and", out_name_gamma)
