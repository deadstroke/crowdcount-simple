# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:29:56 2017

@author: deads

References:
    https://www.learnopencv.com/blob-detection-using-opencv-python-c/
    https://docs.opencv.org/trunk/d8/da7/structcv_1_1SimpleBlobDetector_1_1Params.html
    https://docs.opencv.org/3.2.0/d2/d29/classcv_1_1KeyPoint.html

Download and run to count blobs.
Have the appropriate image files in the directory.
"""

### Standard imports
import cv2
import copy
import numpy as np
import itertools
import random
import datetime
import cPickle as pickle
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from collections import defaultdict
import sklearn, sklearn.model_selection, sklearn.linear_model


#Switches
Correct_Brightness = True
Blow_Greens = True
Suppress_Blacks = True
Compute_PatchDense = True


### Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
# Change thresholds
params.minThreshold = 0 #Darkest blob
params.maxThreshold = 190 #Lightest blob
# Filter by Area.
params.filterByArea = True
params.minArea = 8
params.maxArea = 50
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.35
# Allow close blobs
params.minDistBetweenBlobs = 5.0


### Parameters for color manipulation
ref_intensity = 140.0

ref_green = 170.0
color_range = (30,75) #define greens range
gamma = 0.6 #gamma correction for green -> achieve blow out <1
sig = 15.0 #smoothing kernel for greens

color_range_blacks = (0,130) #define blacks range
gamma_blacks = 1.3 #gamma correction for blacks -> achieve suppression >1
sig_blacks = 15.0 #smoothing kernel for blacks


### Parameters for patch density sampling - by right R should be smaller than the largest gathered crowd
### try R = 20 in future
R = 40; N_Samples = 20000



###############################################################################
### Functions to manipulate colors
###############################################################################
def correct_brightness(im,target):
    def func(gamma,im,target):
        return np.mean(gamma_corr(im,gamma))-target
    
    g0 = fsolve(func, 1.0, args=(im,target))[0]
    return gamma_corr(im,g0)

def gamma_corr(I,gamma):
    return 255.0*(I/255.0)**gamma

def decayed_mux(d):
    return np.exp(-(d/sig)**2 / 2)

def mask_array(H,color_range):
    return (H>color_range[0]) & (H<color_range[1])

def smoothed_gamma_corr_wtarget(H,I,color_range,target,sig=15):
    #Apply mask of color range with H and modify on I
    H = H.astype(int)
    above = H - color_range[1]
    below = color_range[0] - H
    d = np.maximum(above,below)
    d[d<0] = 0
    
    im = I[d==0]
    def func(gamma,im,target):
        return np.mean(gamma_corr(im,gamma))-target
    gamma = fsolve(func, 1.0, args=(im,target))[0]
    
    exponent = decayed_mux(d)  
    applied_gamma = gamma**exponent
    
    return gamma_corr(I,applied_gamma)

def smoothed_gamma_corr(H,I,color_range,gamma,sig=5):
    #Apply mask of color range with H and modify on I
    H = H.astype(int)
    above = H - color_range[1]
    below = color_range[0] - H
    d = np.maximum(above,below)
    d[d<0] = 0
    
    exponent = decayed_mux(d)
    applied_gamma = gamma**exponent
    
    return gamma_corr(I,applied_gamma)



###############################################################################
### Functions to compute patch density
###############################################################################

# Regression corrected density
def corrected_density(X,Coeff,Intercept):
    X = np.array(X)
    X_poly = np.vstack((X,X**2))
    X_poly = X_poly.transpose()
    
    return np.dot(X_poly,Coeff)


# Round window densitities
def round_array(R):
    R_Patch = 2*R + 1
    
    Patch_Arr = np.zeros((R_Patch,R_Patch))
    for r in reversed(range(1,R+1)):
        for x in range(R+1):
            for y in range(R+1):
                if np.linalg.norm([x-R,y-R]) <= r:
                    Patch_Arr[x,y] = r
    
    Patch_Arr[:,R:] = Patch_Arr[:,R::-1]
    Patch_Arr[R:,:] = Patch_Arr[R::-1,:]
    Patch_Arr[R,R] = 1
    Patch_Arr = Patch_Arr>0

    return Patch_Arr

def get_patch(im_mask,x,y,ht,wd):
    patch_mask = np.zeros((ht,wd))
    bounds = (y-max(0,y-R),
              min(ht,y+R)-y,
              x-max(0,x-R),
              min(wd,x+R)-x)
    
    patch_mask[y-bounds[0]:y+bounds[1],x-bounds[2]:x+bounds[3]] = \
    im_mask[y-bounds[0]:y+bounds[1],x-bounds[2]:x+bounds[3]] * \
    Round_Array[R-bounds[0]:R+bounds[1],R-bounds[2]:R+bounds[3]]
    
    return patch_mask



###############################################################################
### Functions to match color histogram - Deprecated
###############################################################################
#
#### Parameters for histogram match, used to use 610pm histogram, its pretty dark, should we change?
#hist_bins = np.array([10, 50, 220, 256])
#region_cumratios = np.array([0.01, 0.99, 1.00]) #610pm values
#
#def get_cumratio(img,img_mask):
#    img_mask = img_mask.astype(bool)
#    masked = img[img_mask]
#    area = float(len(masked))
#    region_count = np.histogram(masked,bins=hist_bins)[0]
#    region_cumratios = np.cumsum(region_count/area)
#    
#    return region_cumratios
#
#
#def ratio_match(img,img_mask):
##    img_blurfloat = cv2.GaussianBlur(img.astype(float),(3,3),0)
#    intensities = img[img_mask]
#    [intensities_sorted, sorted_inds] = zip(*sorted(zip(intensities,range(len(intensities)))))
#    intensities_sorted = np.array(intensities_sorted)
#    transformed_intensities = copy.deepcopy(intensities_sorted)
#    
#    img_area = float(len(intensities))
#    targetcounts = (np.round(np.array(region_cumratios) * img_area)).astype(int)
#    targetcounts = [0] + list(targetcounts)
#    
#    for k,(ind_lower,ind_upper) in enumerate(zip(targetcounts,targetcounts[1:])):
#        int_lower_ref = hist_bins[k]; int_upper_ref = hist_bins[k+1] - 1
#        int_lower = intensities_sorted[ind_lower]
#        int_upper = intensities_sorted[ind_upper-1]
#        
#        #transformations
#        m = (int_upper_ref - int_lower_ref) / float(int_upper - int_lower)
#        
#        temp_intensities = intensities_sorted[ind_lower:ind_upper]
#        temp_intensities = m*(temp_intensities-int_lower) + int_lower_ref
#        transformed_intensities[ind_lower:ind_upper] = temp_intensities
#        
#    transformed_intensities[ind_lower:ind_upper] = np.round(transformed_intensities[ind_lower:ind_upper]).astype(int)
#    [_, transformed_intensities_ordered] = zip(*sorted(zip(sorted_inds,transformed_intensities)))
#    transformed_intensities_ordered = np.array(transformed_intensities_ordered)
#    
#    img_transformed = copy.deepcopy(img)
#    img_transformed[img_mask] = transformed_intensities_ordered
#    img_transformed = np.array(img_transformed, dtype='uint8')
#    
#    return img_transformed

#    # Histogram
#    print 'Original:'
#    print get_cumratio(im,im_mask)
#    im = ratio_match(im,im_mask)
#    print 'Adjusted:'
#    print get_cumratio(im,im_mask)


###############################################################################
### Deprecated Stuff
###############################################################################
#ref_im = cv2.imread('HLP_610pm.jpg',0)
#fig, ax = plt.subplots(1, 1, figsize=(8, 8), sharex=True, sharey=True,
#                         subplot_kw={'adjustable': 'box-forced'})
#ax.imshow(ref_im, interpolation='nearest', cmap='gray')
#ax.set_title('Reference')


#ref_im = cv2.imread('HLP_610pm.jpg',0)
#im_mask = cv2.imread('HLP_610pm_Mask.jpg',0)>127
#masked = ref_im[im_mask]
#area = float(len(masked))
#
#region_count = np.histogram(masked,bins=hist_bins)[0]
#region_cumratios = np.cumsum(region_count/area)


#midlow = 50; midhigh = 220
#
## Use 6.10pm image as the reference midtone level
#ref_im = cv2.imread('HLP_610pm.jpg',0)
#im_mask = cv2.imread('HLP_610pm_Mask.jpg',0)>127
#
#midtones = ref_im[im_mask]
#midtones = midtones[(midlow<midtones)*(midtones<midhigh)]
#ref_midtones = np.mean(midtones) # = 115

#    # Measure midtone brightness and normalize
#    midtones = im[im_mask]
#    midtones = midtones[(midlow<midtones)*(midtones<midhigh)]
#    level_midtones = np.mean(midtones)
#    delta_bright = int(ref_midtones - level_midtones)
#    print "Org midtones: %f" % level_midtones
#    
#    # Normalize Brightness
#    im = im.astype(int)
#    im = im + delta_bright
#    im[im<0] = 0; im[im>255] = 255
#    im = np.array(im, dtype='uint8')




#
#img = cv2.imread('HLP_630pm.jpg',0)
#img_mask = cv2.imread('HLP_630pm_Mask.jpg',0)>127
#img_transformed = ratio_match(img,img_mask)
##
#fig, ax = plt.subplots(1, 1, figsize=(8, 8), sharex=True, sharey=True,
#                         subplot_kw={'adjustable': 'box-forced'})
#ax.imshow(ref_im, interpolation='nearest', cmap='gray')
#ax.set_title('Reference')
#
#fig, ax = plt.subplots(1, 1, figsize=(8, 8), sharex=True, sharey=True,
#                         subplot_kw={'adjustable': 'box-forced'})
#ax.imshow(cv2.GaussianBlur(ref_im.astype(float),(21,21),0), interpolation='nearest', cmap='gray')
#ax.set_title('Blur')
#

#
#fig, ax = plt.subplots(1, 1, figsize=(8, 8), sharex=True, sharey=True,
#                         subplot_kw={'adjustable': 'box-forced'})
#ax.imshow(img, interpolation='nearest', cmap='gray')
#ax.set_title('Original')
#
#fig, ax = plt.subplots(1, 1, figsize=(8, 8), sharex=True, sharey=True,
#                         subplot_kw={'adjustable': 'box-forced'})
#ax.imshow(img_transformed, interpolation='nearest', cmap='gray')
#ax.set_title('Transformed')
#
#region_cumratios
#get_cumratio(img,img_mask)
#get_cumratio(img_transformed,img_mask)




#    # Density scatter
#    fig = plt.figure(figsize=(8, 8))
#    x,y = zip(*[tup for lst in Density_TupsAll for tup in lst])
#    plt.scatter(x,y,marker='.')
#    plt.plot([-1000,1000],[-1000,1000],color='r')
#    plt.axis('equal')
#    plt_min = min(min(x),min(y))
#    plt_max = max(max(x),max(y))
#    plt.xlim(plt_min,plt_max)
#    plt.ylim(plt_min,plt_max)
#    plt.title('Density Scatter')
#    plt.xlabel('Blob Density')
#    plt.ylabel('Truth Density')
#    plt.show()
#    
#    
#    import datetime
#    dt_str = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
#    pickle.dump(Density_TupsAll,open('DensityAll_'+dt_str+'.p', "wb"))
    
    
    
    
    
    




