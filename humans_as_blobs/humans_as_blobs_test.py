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

Integral_Count = True

### Common initialization
execfile('humans_as_blobs_common.py')

### Use regressor from
(R,N_Samples,Intercept,Coeff,bins,Top,Bottom) = \
pickle.load(open("2017-11-09_10.26.49_Regression_n_Error.p", "rb"))
N_Samples = N_Samples #Want to use smaller sample?
N_Samples = 2500


### Read files - read as HSV!!!
im_all = [cv2.imread('440pm.jpg'),]
im_mask_all = [cv2.imread('mask.jpg',0)>127,]
for i,im in enumerate(im_all):
    im_all[i] = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

### Scaling - seems best to not fool with it
scale = 1.0 # original image humans 18 px tall
R = int(R*scale)
params.minArea = params.minArea*scale
params.maxArea = params.maxArea*scale
params.minDistBetweenBlobs = params.minDistBetweenBlobs*scale

### Compute binary circle array
Round_Array = round_array(R)
Density_TupsAll = [] #(blob_dense,actual_dense)

### Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
    detector = cv2.SimpleBlobDetector(params)
else : 
    detector = cv2.SimpleBlobDetector_create(params)
    

### Detection procedure
for k,(im,im_mask) in enumerate(zip(im_all,im_mask_all)):
    print "\nImage %d" % k
    
    if Correct_Brightness:
        print "Correct brightness"
        #print "Original Intensity: %.2f" % np.mean(im[:,:,2])
        im[:,:,2] = correct_brightness(im[:,:,2],ref_intensity)
        #print "Corrected Intensity: %.2f" % np.mean(im[:,:,2])    
    if Blow_Greens:
        H = im[:,:,0]
        I = im[:,:,2]
        I_new = smoothed_gamma_corr(H,I,color_range,gamma,sig=sig)
#        I_new = smoothed_gamma_corr_wtarget(H,I,color_range,ref_green,sig=sig)
        im[:,:,2] = I_new
        print "Blow out green luminence"
        #print "Green luminence: %.2f" % np.mean(I_new[mask_array(H,color_range)])
    if Suppress_Blacks:
        im[:,:,2] = smoothed_gamma_corr(im[:,:,2],im[:,:,2],color_range_blacks,gamma_blacks,sig=sig_blacks)
        print "Suppress blacks"
        I = im[:,:,2]
        #print "Black luminence: %.2f" % np.mean(I[mask_array(I,color_range_blacks)])
    
    # To grayscale
    im_gray = im[:,:,2]
    
    # Detect humans with tuned detector, filter out blobs in mask
    keypoints = detector.detect(im_gray)
    Blobs_InMask = [(kp.pt[1],kp.pt[0]) for kp in keypoints if im_mask[int(kp.pt[1]),int(kp.pt[0])]]
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), sharex=True, sharey=True,
                             subplot_kw={'adjustable': 'box-forced'})
    coords = Blobs_InMask; color = 'red'; title = 'Predicted'
    ax.imshow(im_gray, interpolation='nearest', cmap='gray')
    ax.set_title(title+' Count: '+str(len(coords)))
    for (y,x) in coords:
        c = plt.Circle((x, y), 1, color=color, linewidth=1, fill=False)
        ax.add_patch(c)
    plt.tight_layout()
    plt.show()
    
    # Compute densities
    if Integral_Count:
        ht,wd = im_mask.shape
        mask_coords = np.where(im_mask)
        mask_coords = zip(mask_coords[0],mask_coords[1])
        random.shuffle(mask_coords)
        Density = []
        for i,n in enumerate(range(N_Samples)):
            (y,x) = mask_coords[i]
            patch_mask = get_patch(im_mask,x,y,ht,wd)
            Blobs_InPatch = [(kp.pt[1],kp.pt[0]) for kp in keypoints if patch_mask[int(kp.pt[1]),int(kp.pt[0])]]
            Area = float(sum(patch_mask.ravel()))
            Blob_count = len(Blobs_InPatch)
            Density += [Blob_count/Area]
            
        area = sum(im_mask.ravel()>0)
        int_counts = np.mean(Density)*area
        
        # Compute corrected counts
        print "Integrated blob density: %d" % int(int_counts)
        print "Integrated blob density (with correction): %d" % \
        int(np.mean(corrected_density(Density,Coeff,Intercept))*area)
        
    #Error bounds
    bin_int = np.digitize(len(coords), bins)-1
    print "75th percentile bound: %d" % Top[bin_int]
    print "25th percentile bound: %d" % Bottom[bin_int]
    















