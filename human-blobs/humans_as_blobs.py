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

# Standard imports
import cv2
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
midlow = 50; midhigh = 220

# Read files
im_all = [cv2.imread('HLP_610pm.jpg',0),
          cv2.imread('HLP_630pm.jpg',0),
          cv2.imread('HLP_640pm.jpg',0)]
im_mask_all = [cv2.imread('HLP_610pm_Mask.jpg',0),
               cv2.imread('HLP_630pm_Mask.jpg',0),
               cv2.imread('HLP_640pm_Mask.jpg',0)]
centroid_list_all = [pickle.load(open("HLP_610pm - 788v780 Coords.p", "rb")),
                     pickle.load(open("HLP_630pm - 513v509 Coords.p", "rb")),
                     pickle.load(open("HLP_640pm - 216v213 Coords.p", "rb"))]


# Use 6.10pm image as the reference midtone level
im = im_all[0]; im_mask = im_mask_all[0]
midtones = im[im_mask]
midtones = midtones[(midlow<midtones)*(midtones<midhigh)]
ref_midtones = np.mean(midtones)


# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
# Change thresholds
params.minThreshold = 50
# Filter by Area.
params.filterByArea = True
params.minArea = 10
params.maxArea = 1200
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.35
# Allow close blobs
params.minDistBetweenBlobs = 5.0

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
    detector = cv2.SimpleBlobDetector(params)
else : 
    detector = cv2.SimpleBlobDetector_create(params)
    

for k,(im,im_mask,centroid_list) in enumerate(zip(im_all,im_mask_all,centroid_list_all)):
    
    # Measure midtone brightness and normalize
    im_mask = im_mask>127
    midtones = im[im_mask]
    midtones = midtones[(midlow<midtones)*(midtones<midhigh)]
    level_midtones = np.mean(midtones)
    delta_bright = int(ref_midtones - level_midtones)
    print "Org midtones: %f" % level_midtones
    
    # Normalize Brightness
    im = im.astype(int)
    im = im + delta_bright
    im[im<0] = 0; im[im>255] = 255
    im = np.array(im, dtype='uint8')
    
    #Verify tuning
    midtones = im[im_mask]
    midtones = midtones[(midlow<midtones)*(midtones<midhigh)]
    level_midtones = np.mean(midtones)
    print "Adjusted midtones: %f" % level_midtones
    
    # Detect humans with tuned detector
    keypoints = detector.detect(im)
    Blobs_InMask = [(kp.pt[1],kp.pt[0]) for kp in keypoints if im_mask[int(kp.pt[1]),int(kp.pt[0])]]
    Marked_InMask = [(y,x) for (y,x) in centroid_list if im_mask[int(y),int(x)]]
    
    # Plot
    # np.mean(im)
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharex=True, sharey=True,
                             subplot_kw={'adjustable': 'box-forced'})
    ax = axes.ravel()
    for i,(coords,color,title) in enumerate(zip([Blobs_InMask, Marked_InMask],['red','red'],['Predicted','Ground Truth'])):
        ax[i].imshow(im, interpolation='nearest', cmap='gray')
        ax[i].set_title(title+' Count: '+str(len(coords)))
        for (y,x) in coords:
            c = plt.Circle((x, y), 1, color=color, linewidth=1, fill=False)
            ax[i].add_patch(c)
    
    plt.tight_layout()
    plt.show()


















