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

### To pickle or not to pickle
Pickle_Densities = True

### Common initialization
execfile('humans_as_blobs_common.py')

### Read files - read as HSV!!!
im_all = [cv2.imread('HLP_610pm.jpg'),
          cv2.imread('HLP_630pm.jpg'),
          cv2.imread('HLP_640pm.jpg')]
im_mask_all = [cv2.imread('HLP_610pm_Mask.jpg',0)>127,
               cv2.imread('HLP_630pm_Mask.jpg',0)>127,
               cv2.imread('HLP_640pm_Mask.jpg',0)>127]
centroid_list_all = [pickle.load(open("HLP_610pm - 788v780 Coords.p", "rb")),
                     pickle.load(open("HLP_630pm - 513v509 Coords.p", "rb")),
                     pickle.load(open("HLP_640pm - 216v213 Coords.p", "rb"))]
for i,im in enumerate(im_all):
    im_all[i] = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

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
for k,(im,im_mask,centroid_list) in enumerate(zip(im_all,im_mask_all,centroid_list_all)):
    print "\nImage %d" % k
    
    if Correct_Brightness:
        print "Correct brightness"
        #print "Original Intensity: %.2f" % np.mean(im[:,:,2])
        im[:,:,2] = correct_brightness(im[:,:,2],ref_intensity)
        #print "Corrected Intensity: %.2f" % np.mean(im[:,:,2])    
    if Blow_Greens:
        H = im[:,:,0]
        I = im[:,:,2]
        #I_new = smoothed_gamma_corr(H,I,color_range,gamma,sig=sig)
        I_new = smoothed_gamma_corr_wtarget(H,I,color_range,ref_green,sig=sig)
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
    Marked_InMask = [(y,x) for (y,x) in centroid_list if im_mask[int(y),int(x)]]    
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharex=True, sharey=True,subplot_kw={'adjustable': 'box-forced'})
    ax = axes.ravel()
    for i,(coords,color,title) in enumerate(zip([Blobs_InMask, Marked_InMask],['red','red'],['Predicted','Ground Truth'])):
        ax[i].imshow(im_gray, interpolation='nearest', cmap='gray')
        #ax[i].imshow(cv2.cvtColor(im, cv2.COLOR_HSV2RGB))
        ax[i].set_title(title+' Count: '+str(len(coords))) #+' Integral Count: '+str(int(int_counts[i])))
        for (y,x) in coords:
            c = plt.Circle((x, y), 1, color=color, linewidth=1, fill=False)
            ax[i].add_patch(c)
    
    plt.tight_layout()
    plt.show()
    
    # Compute patch densities
    if Compute_PatchDense:
        ht,wd = im_mask.shape
        mask_coords = np.where(im_mask)
        mask_coords = zip(mask_coords[0],mask_coords[1])
        random.shuffle(mask_coords)
        Density_Tups = []
        for i,n in enumerate(range(N_Samples)):
            (y,x) = mask_coords[i]
            patch_mask = get_patch(im_mask,x,y,ht,wd)
            
            Blobs_InPatch = [(kp.pt[1],kp.pt[0]) for kp in keypoints if patch_mask[int(kp.pt[1]),int(kp.pt[0])]]
            Marked_InPatch = [(y,x) for (y,x) in centroid_list if patch_mask[int(y),int(x)]]
            Area = float(sum(patch_mask.ravel()))
            
            Blob_count = len(Blobs_InPatch)
            Actual_count = len(Marked_InPatch)
            Density_Tups += [(Blob_count/Area, Actual_count/Area)]
            
        Density_TupsAll += [Density_Tups]
        area = sum(im_mask.ravel()>0)
        pred,actual = zip(*Density_Tups)
        int_counts = (np.mean(pred)*area,np.mean(actual)*area)
        
        print "Integrated blob density: %d" % int_counts[0]


### Regression correction for patch densities
if Compute_PatchDense:
    Density_TupsFlat = [tup for lst in Density_TupsAll for tup in lst]
    X,y = zip(*Density_TupsFlat)
    X = np.array(X); y = np.array(y)
    
    # Regression modelling - assume polynomials x and x^2, c = 0
    X_poly = np.vstack((X,X**2))
    X_poly = X_poly.transpose()
    model = sklearn.linear_model.LinearRegression(fit_intercept=False)
    #model = sklearn.linear_model.LinearRegression(fit_intercept=True)
    model.fit(X_poly,y)
    Coeff = model.coef_.reshape((X_poly.shape[1],1))
    Intercept = model.intercept_
    
    # Plotting
    fig = plt.figure(figsize=(8, 8))
    plt_min = min(min(X),min(y))
    plt_max = max(max(X),max(y))
    plt.scatter(X,y,marker='.') #Scatter
    plt.plot([-1000,1000],[-1000,1000],color='r') #y=x line
    # Plot regression
    X_plot = np.linspace(plt_min,plt_max)
    X_plot_poly = np.vstack((X_plot,X_plot**2)).transpose()
    plt.plot(X_plot,np.dot(X_plot_poly,Coeff)+Intercept,linestyle=':',color='yellow',linewidth=4.0)
    
    plt.axis('equal')
    plt.xlim(plt_min,plt_max)
    plt.ylim(plt_min,plt_max)
    plt.title('Density Scatter')
    plt.xlabel('Blob Density')
    plt.ylabel('Truth Density')
    plt.show()
    
    # Print coefficients
    print "\nRegression coefficients:"
    print "Intercept:\t\t%.3f" % Intercept
    print "X Coeff:\t\t%.3f" % Coeff[0][0]
    print "X**2 Coeff:\t\t%.3f" % Coeff[1][0]
    
    # Error bounds
    area = 285000 #archtypical area
    X_count = area*X
    y_count = area*y
    B = 25 #Error bins
    bins = np.linspace(0,max(X_count),B)
    bin_digit = np.digitize(X_count, bins)
    
    # Throw y points to bins
    count_dict = defaultdict(list)
    for i,d in enumerate(bin_digit):
        count_dict[d] += [y_count[i]]
    Top = [np.percentile(count_dict[d],75) for d in range(1,B)]
    Bottom = [np.percentile(count_dict[d],25) for d in range(1,B)]
    bin_mid = (bins[:-1]+bins[1:])/2
    
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(X_count,y_count,marker='.')
    plt.plot([-5000,5000],[-5000,5000],color='r')
    plt.plot(bin_mid, Top, linestyle=':',color='orange',linewidth=4.0)
    plt.plot(bin_mid, Bottom, linestyle=':',color='orange',linewidth=4.0)
    
    plt.axis('equal')
    plt_min = min(min(X_count),min(y_count))
    plt_max = max(max(X_count),max(y_count))
    plt.xlim(plt_min,plt_max)
    plt.ylim(plt_min,plt_max)
    plt.title('Error Bounds')
    plt.xlabel('Blob Count')
    plt.ylabel('Truth Count')
    plt.show()

    # Compute corrected counts
    print "\nComputing correction counts:"
    for i in range(len(im_mask_all)):
        im_mask = im_mask_all[i]
        area = sum(im_mask.ravel())
        X,y = zip(*Density_TupsAll[i])
        X = np.array(X); y = np.array(y)
        X_poly = np.vstack((X,X**2))
        X_poly = X_poly.transpose()
        X_corrected = np.dot(X_poly,Coeff)
        
        print "Image %d" % i
        print "Integrated blob density: %d" % int(np.mean(X)*area)
        print "Integrated blob density (with correction): %d" % int(np.mean(X_corrected)*area)
        
    # Write density data
    if Pickle_Densities:
        dt_str = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
        Pickle_Tup = (R,N_Samples,Intercept,Coeff,bins,Top,Bottom)
        
        pickle.dump(Density_TupsAll,open(dt_str+'_DensityAll'+'.p', "wb"))
        pickle.dump(Pickle_Tup,open(dt_str+'_Regression_n_Error'+'.p', "wb"))

    

























