
""" This is a module for hdr imaging homework (15-463/663/862, Computational Photography, Fall 2020, CMU).

Modified by cmetzler on 9/13/2022

You can import necessary functions into your code as follows:
from HDR_helperfiles import *

Depends on OpenCV to read/write HDR files"""

import numpy as np
import cv2
import exifread
from PIL import Image
import os

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


 # parts 3 and 4

def reinhard_tonemap(hdr, K=0.15, B=0.95, epsilon=1e-6): # impliment formula
    
    luminance = np.mean(hdr, axis=2) + epsilon
    log_luminance = np.log(luminance)
    N = np.prod(luminance.shape)
    avg_log_luminance = np.exp(np.sum(log_luminance) / N)
    
    scaled_hdr = (K / avg_log_luminance) * hdr
    max_val = np.max(scaled_hdr)
    I_white = B * max_val
    
    numerator = scaled_hdr * (1.0 + (scaled_hdr / (I_white**2 + epsilon)))
    denominator = 1.0 + scaled_hdr
    tone_mapped = numerator / (denominator + epsilon)
    
    
    tone_mapped = np.clip(tone_mapped, 0.0, 1.0)
    return tone_mapped

def gamma_srgb(img_linear):
    
    mask = (img_linear <= 0.0031308)
    low  = 12.92 * img_linear
    high = 1.055 * (img_linear ** (1.0 / 2.5)) - 0.055
    return np.where(mask, low, high)




K_values = [0.01,0.05,0.15,0.5,1] # create arrays of k and b to iterate over
B_values = [0.01,0.05,0.1,0.5,1] 

# iterate k and b over the gaussian weight

hdr_img = HDR_images[2]  
    
    # display and save both results with and without gamma correction
for K_val in K_values:
    for B_val in B_values:
            
            
        tonemapped_img = reinhard_tonemap(hdr_img, K=K_val, B=B_val)
            
            
        tonemapped_img_gamma = gamma_srgb(tonemapped_img)
        tonemapped_img_gamma = np.clip(tonemapped_img_gamma, 0.0, 1.0)
            
            
        plt.figure()
        plt.title(f"Gaussian, Tone-Mapped Only (K={K_val:.2f}, B={B_val:.2f})")
        plt.imshow(tonemapped_img)
        plt.show()
            
            
        plt.figure()
        plt.title(f"Gaussian, Tone-Mapped + Gamma (K={K_val:.2f}, B={B_val:.2f})")
        plt.imshow(tonemapped_img_gamma)
        plt.show()
            
            
        out_name_nogamma = f"Gaussian_K{K_val:.2f}_B{B_val:.2f}_nogamma.png"
        writeHDR(out_name_nogamma, tonemapped_img)
            
            
        out_name_gamma = f"Gaussian_K{K_val:.2f}_B{B_val:.2f}_gamma.png"
        writeHDR(out_name_gamma, tonemapped_img_gamma)

        print("Saved:", out_name_nogamma, "and", out_name_gamma)


# part 5

hdr_img = HDR_images[2]  # pick second weight and the k and b value recommended in the pdf

tonemapped_img = reinhard_tonemap(hdr_img, K=0.15, B=0.95)


tonemapped_img_gamma = gamma_srgb(tonemapped_img)
tonemapped_img_gamma = np.clip(tonemapped_img_gamma, 0.0, 1.0)

selected_img = tonemapped_img_gamma  


png_filename = "final_output.png"
writeHDR(png_filename, selected_img)  
print(f"Saved lossless PNG: {png_filename}")


png_size = os.path.getsize(png_filename)
print(f"PNG file size: {png_size} bytes")


ldr_8bit = (selected_img * 255).astype(np.uint8) # convert to bgr for opencv
ldr_bgr = ldr_8bit[..., ::-1]  


qualities = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10] # jpeg qualities to sweep through and compare to the png


for Q in qualities:
    
    jpeg_filename = f"final_output_Q{Q}.jpg"
    cv2.imwrite(jpeg_filename, ldr_bgr, [cv2.IMWRITE_JPEG_QUALITY, Q])
    
    
    jpeg_size = os.path.getsize(jpeg_filename)
    
    
    if jpeg_size > 0:
        compression_ratio = png_size / jpeg_size
    else:
        compression_ratio = 0 # make sure we dont divide by 0 incase image is empty
    
    
    print(f"  Quality = {Q:3d},  JPEG Size = {jpeg_size:6d} bytes,  "
          f"Compression Ratio = {compression_ratio:.2f}")
    
    
    jpeg_img_bgr = cv2.imread(jpeg_filename, cv2.IMREAD_UNCHANGED)
    jpeg_img_rgb = jpeg_img_bgr[..., ::-1]
    plt.figure()
    plt.title(f"JPEG Quality {Q}")
    plt.imshow(jpeg_img_rgb)
    plt.axis('off')
    plt.show()
