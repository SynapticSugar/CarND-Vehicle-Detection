import numpy as np
import os
import glob
import pickle
import cv2
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import lane_finder as lf
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label

def convert_color(img, conv='RGB2YCrCb'):
    '''
    Utility to convert to various color spaces.
    Adapted from Udacity Vehicle Detection and Tracking Project Lesson
    '''
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    '''
    Method to return HOG features and visualisation
    Adapted from Udacity Vehicle Detection and Tracking Project Lesson
    '''
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, block_norm='L2-Hys',
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, block_norm='L2-Hys',
                       visualise=vis, feature_vector=feature_vec)
        return features

def bin_spatial(img, size=(32, 32)):
    '''
    Method to compute binned color features.
    Adapted from Udacity Vehicle Detection and Tracking Project Lesson
    '''
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))

def color_hist(img, nbins=32):    #bins_range=(0, 256)
    '''
    Method to compute color histogram features.
    Adapted from Udacity Vehicle Detection and Tracking Project Lesson
    '''
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    '''
    Utility to draw bounding boxes.
    Adapted from Udacity Vehicle Detection and Tracking Project Lesson
    '''
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates

        cv2.rectangle(imcopy, (int(bbox[0][0]),int(bbox[0][1])), (int(bbox[1][0]),int(bbox[1][1])), color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    '''
    Method to extract features from a list of images. Takes file names of images.
    Adapted from Udacity Vehicle Detection and Tracking Project Lesson
    '''
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    '''
    Method to generate a list of bounding box windows.
    Adapted from Udacity Vehicle Detection and Tracking Project Lesson
    '''
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True, visualise=False):
    '''
    Method to extract features from a single image window.
    Adapted from Udacity Vehicle Detection and Tracking Project Lesson
    '''
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
        else:
            if visualise:
                hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                    pix_per_cell, cell_per_block, vis=True, feature_vec=True)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                    pix_per_cell, cell_per_block, vis=False, feature_vec=True)

        #8) Append features to list
        img_features.append(hog_features)
    #9) Return concatenated array of features
    if visualise:
        return np.concatenate(img_features), hog_image, spatial_features, hist_features
    else:
        return np.concatenate(img_features)

def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):
    '''
    Method to search an image  with the given list of windows.
    Adapted from Udacity Vehicle Detection and Tracking Project Lesson
    '''
    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

def add_heat(heatmap, bbox_list):
    '''
    Adapted from Udacity Vehicle Detection and Tracking Project Lesson
    '''
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    '''
    Adapted from Udacity Vehicle Detection and Tracking Project Lesson
    '''
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels, min_size=(50,50)):
    '''
    Adapted from Udacity Vehicle Detection and Tracking Project Lesson
    '''
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        if ((abs(bbox[1][0]-bbox[0][0]) > min_size[0]) & 
            (abs(bbox[1][1]-bbox[0][1]) > min_size[1])):
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
            cv2.putText(img, str(car_number), bbox[0], cv2.FONT_HERSHEY_SIMPLEX, 0.75, [255,255,255], 2, 8)
    # Return the image
    return img

def get_training_set():
    '''
    Utility to read the training set file names.
    '''
    cars = []
    notcars = []
    # load folder names for cars and noncars
    root = 'vehicles/'
    db_types = os.listdir(root)
    for db_type in db_types:
        cars.extend(glob.glob(root+db_type+'/*'))
    root = 'non-vehicles/'
    db_types = os.listdir(root)
    for db_type in db_types:
        notcars.extend(glob.glob(root+db_type+'/*'))
    print("found {} cars and {} notcars".format(len(cars), len(notcars)))
    return cars, notcars

def visualise_plot(fig, rows, cols, images, titles, is_plot=False):
    '''
    Utility to plot visualisation.
    From https://www.youtube.com/watch?v=P2zwrTM8ueA&index=5&list=PLAwxTw4SYaPkz3HerxrHlu1Seq8ZA7-5P
    '''
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i+1)
        plt.title(i+1)
        img_dims = len(img.shape)
        if img_dims == 1:
            if is_plot:
                plt.plot(img)
            else:
                print(len(img))
                plt.bar(range(0,len(img),1),img[0:img.shape[0]])
        if img_dims == 2:
            plt.imshow(img, cmap='bone')
        if img_dims > 2:
            plt.imshow(img)
        plt.title(titles[i])
    plt.tight_layout()

def visualise_hog(colorspace='RGB'):
    '''
    Utility to visulize the HOG features.
    Adapted from https://www.youtube.com/watch?v=P2zwrTM8ueA&index=5&list=PLAwxTw4SYaPkz3HerxrHlu1Seq8ZA7-5P
    '''
    # chose a random training set
    car_ind =4133# np.random.randint(0, len(cars))#5575#
    print(car_ind)
    notcar_ind = 4133#np.random.randint(0, len(notcars))
    # read images
    car_image = mpimg.imread(cars[car_ind])
    notcar_image = mpimg.imread(notcars[notcar_ind])
    cf = []
    chi = []
    ncf = []
    nchi = []
    for channel in range(3):
        car_features, car_hog_image, car_spatial_features, car_histogram_features = single_img_features(car_image, color_space=colorspace,
                                spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, 
                                pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                                hog_channel=channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat, visualise=True)
        notcar_features, notcar_hog_image, notcar_spatial_features, notcar_histogram_features = single_img_features(notcar_image, color_space=colorspace,
                                spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, 
                                pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                                hog_channel=channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat, visualise=True)
        cf.append(car_features)
        chi.append(car_hog_image)
        csf = car_spatial_features
        chf = car_histogram_features
        ncf.append(notcar_features)
        nchi.append(notcar_hog_image)
        ncsf = notcar_spatial_features
        nchf = notcar_histogram_features

    # HOG
    images = [car_image, chi[0], chi[1], chi[2], notcar_image, nchi[0], nchi[1], nchi[2]]
    titles = ['car image', 'HOG '+colorspace+'[0]', 'HOG '+colorspace+'[1]', 'HOG '+colorspace+'[2]',
              'notcar image', 'HOG '+colorspace+'[0]', 'HOG '+colorspace+'[1]', 'HOG '+colorspace+'[2]']
    fig = plt.figure(figsize=(12,6))
    visualise_plot(fig, 2, 4, images, titles)
    plt.savefig('output_images/visualise_{}_HOG_features.png'.format(colorspace))
    # plt.show()
    # Spatial
    images = [car_image, csf, notcar_image, ncsf]
    titles = ['car image', ' Spacial '+colorspace,
              'notcar image', 'Spacial '+colorspace]
    fig = plt.figure(figsize=(12,6))
    visualise_plot(fig, 2, 2, images, titles, is_plot=True)
    plt.savefig('output_images/visualise_{}_spatial_features.png'.format(colorspace))
    # plt.show()
    # Histogram
    images = [car_image, chf,  notcar_image, nchf]
    titles = ['car image', ' Hist '+colorspace,
              'notcar image', 'Hist '+colorspace]
    fig = plt.figure(figsize=(12,6))
    visualise_plot(fig, 2, 2, images, titles)
    plt.savefig('output_images/visualise_{}_hist_features.png'.format(colorspace))
    # plt.show()

def visualise_spacial_binning(color_space='RGB'):
    '''
    Utility to visualise a color space binning.
    '''
    # chose a random training set
    car_ind = 99#np.random.randint(0, len(cars))
    notcar_ind = 99#np.random.randint(0, len(notcars))
    # read images
    car_image = mpimg.imread(cars[car_ind])
    notcar_image = mpimg.imread(notcars[car_ind])
    car_spatial_image = car_image.copy()
    notcar_spatial_image = notcar_image.copy()
    if color_space != 'RGB':
        pass
    elif color_space == 'HSV':
        car_spatial_image = cv2.cvtColor(car_image, cv2.COLOR_RGB2HSV)
        notcar_spatial_image = cv2.cvtColor(notcar_image, cv2.COLOR_RGB2HSV)
    elif color_space == 'LUV':
        car_spatial_image = cv2.cvtColor(car_image, cv2.COLOR_RGB2LUV)
        notcar_spatial_image = cv2.cvtColor(notcar_image, cv2.COLOR_RGB2LUV)
    elif color_space == 'HLS':
        car_spatial_image = cv2.cvtColor(car_image, cv2.COLOR_RGB2HLS)
        notcar_spatial_image = cv2.cvtColor(notcar_image, cv2.COLOR_RGB2HLS)
    elif color_space == 'YUV':
        car_spatial_image = cv2.cvtColor(car_image, cv2.COLOR_RGB2YUV)
        notcar_spatial_image = cv2.cvtColor(notcar_image, cv2.COLOR_RGB2YUV)
    elif color_space == 'YCrCb':
        car_spatial_image = cv2.cvtColor(car_image, cv2.COLOR_RGB2YCrCb)
        notcar_spatial_image = cv2.cvtColor(notcar_image, cv2.COLOR_RGB2YCrCb)

    color1 = cv2.resize(car_spatial_image[:,:,0], spatial_size)
    color2 = cv2.resize(car_spatial_image[:,:,1], spatial_size)
    color3 = cv2.resize(car_spatial_image[:,:,2], spatial_size)
    ncolor1 = cv2.resize(notcar_spatial_image[:,:,0], spatial_size)
    ncolor2 = cv2.resize(notcar_spatial_image[:,:,1], spatial_size)
    ncolor3 = cv2.resize(notcar_spatial_image[:,:,2], spatial_size)
    images = [car_image, color1, color2, color3, notcar_image, ncolor1, ncolor2, ncolor3]
    titles = ['car image', color_space+' channel 0', color_space+' channel 1', color_space+' channel 2',
    'notcar image', color_space+' channel 0', color_space+' channel 1', color_space+' channel 2']
    fig = plt.figure(figsize=(12,6))
    visualise(fig, 2, 4, images, titles)
    plt.savefig('output_images/visualise_spatial_features.png')
    # plt.show()

def trainLinearClassifier(n_samples=0):
    '''
    Train a linear SVN classifier to classify 'vehicle' 'non-vehicle'
    Adapted from lecture notes and the Vehicle Tracking Q&A video: https://www.youtube.com/watch?v=P2zwrTM8ueA
    '''
    t = time.time()
    if n_samples:
        random_idxs = np.random.randint(0, len(cars), n_samples)
        test_cars = np.array(cars)[random_idxs]
        test_notcars = np.array(notcars)[random_idxs]
    else:
        test_cars = cars
        test_notcars = notcars

    print('{},{} Training samples...'.format(len(test_cars),len(test_notcars)))

    car_features = extract_features(test_cars, color_space=colorspace, spatial_size=spatial_size,
                            hist_bins=hist_bins, orient=orient, 
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                            spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(test_notcars, color_space=colorspace, spatial_size=spatial_size,
                            hist_bins=hist_bins, orient=orient, 
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                            spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

    print('{:0.02f}Seconds to compute features...'.format(time.time()-t))

    # Stack the X features, one feature per row
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Scale the values 
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)

    # Array of labels match the indies in X
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split( scaled_X, y, test_size=0.1, random_state=rand_state)

    print('Using:',colorspace,'color space,', orient,'orientations,',pix_per_cell,
        'pix_per_cell,',cell_per_block,'cells_per_block,',
        hist_bins, 'hist_bins, and',spatial_size,'spatial sampling')
    print('Feature vector length:', len(X_train[0]))

    # linear SVM
    svc = LinearSVC()

    t=time.time()
    svc.fit(X_train, y_train)
    print('{:0.02f} Seconds to train SVC...'.format(time.time()-t))

    # Check accuracy
    print('TestAccuracy of SVC = {:0.04f}'.format(svc.score(X_test,y_test)))

    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

    return svc, X_scaler

def trackImages(path):
    '''
    A method to open and image, detect cars and draw the elementary detection windows.
    '''
    example_images = glob.glob(path)
    images = []
    titles = []
    y_start_stop = [0, 720]
    overlap = 0.5
    for img_src in example_images:
        t1 = time.time()
        img = mpimg.imread(img_src)
        draw_img = np.copy(img)
        img = img.astype(np.float32)/255
        # make scale sure is between 0 and 1 obvious
        print(np.min(img), np.max(img))

        windows = slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop,
                    xy_window=(96,96), xy_overlap=(overlap, overlap))
        hot_windows = search_windows(img, windows, svc, X_scaler, color_space=colorspace,
                    spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, 
                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                    hog_channel=hog_channel, spatial_feat=spatial_feat, 
                    hist_feat=hist_feat, hog_feat=hog_feat)

        window_img = draw_boxes(draw_img, hot_windows, color=(0,0,255), thick=6)
        images.append(window_img)
        titles.append('')
        print(time.time()-t1, 'Seconds to process one image searching', len(windows),'windows')
    fig = plt.figsize=(12,10)#, dpi=300)
    visualise_plot(fig, 5, 2, images, titles)
    #plt.imshow(window_img)
    plt.show()

def saveLinearClassifier(fname, svc, X_scaler):
    '''
    Method to save the linear SVM classifier to file.
    '''
    pickle_file = {}
    pickle_file = {}
    pickle_file["svc"] = svc
    pickle_file["X_scaler"] = X_scaler
    pickle.dump(pickle_file, open( fname, "wb" ))

def loadLinearClassifier(fname):
    '''
    Method to load the linear SVM classifier from file.
    '''
    if os.path.isfile(fname):
        pickle_file = pickle.load(open( fname, "rb" ))
        return pickle_file["svc"], pickle_file["X_scaler"]
    else:
        return 0,0

def find_cars(img, ystart, ystop, xstart, xstop, scale, svc, X_scaler, orient,
    pix_per_cell, cell_per_block, spatial_size, hist_bins):
    '''
    Method that can extract features using hog sub-sampling and make predictions.
    Adapted from Udacity Vehicle Detection and Tracking Project Lesson
    '''
    img_boxes = []
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    heatmap = np.zeros_like(img[:,:,0])
    img_tosearch = img[ystart:ystop,xstart:xstop]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale),
            np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    count = 0
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, 
                hist_features, hog_features)).reshape(1, -1))
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img, (xbox_left+xstart, ytop_draw+ystart),
                    (xbox_left+xstart+win_draw, ytop_draw+ystart+win_draw),(0,0,255), thickness=6)
                img_boxes.append((((xbox_left+xstart, ytop_draw+ystart),
                    (xbox_left+xstart+win_draw, ytop_draw+ystart+win_draw))))
                heatmap[ytop_draw+ystart:ytop_draw+win_draw+ystart,
                    xbox_left+xstart:xbox_left+xstart+win_draw] += 1
            count += 1

    return draw_img, heatmap, img_boxes, count

def trackCars(path):
    '''
    Utility to track cars in an image set. Used for verification.
    '''
    example_images = glob.glob(path)
    ystart = 400
    ystop = 720
    xstart = 0
    xstop = 1280
    scale = 1.0
    out_images = []
    out_titles = []
    out_maps = []
    out_boxes = []
    for img_src in example_images:
        t1 = time.time()
        img = mpimg.imread(img_src)
        draw_img, heatmap, img_boxes, _ = find_cars(img, ystart, ystop, xstart, 
            ystartscale, svc, X_scaler, orient, pix_per_cell, cell_per_block,
            spatial_size, hist_bins)
        heatmap = apply_threshold(heatmap, 3)
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(img), labels)
        out_images.append(draw_img)
        out_titles.append(img_src[-9:])
        out_titles.append(img_src[-9:])
        out_images.append(heatmap)
        out_maps.append(heatmap)
        out_boxes.append(img_boxes)
        print(time.time()-t1, 'seconds to run')
    fig = plt.figure(figsize=(12,14))
    visualise_plot(fig, 6, 2, out_images, out_titles)
    plt.show()



def trackZoomHeatmap(img):
    '''
    Method to track cars in heatmap for a range of scales.
    '''
    img1 = img.astype(np.float32)/255
    heat_map = np.zeros_like(img1[:,:,0])
    total = 0
    for i in range(track.scales):
        _, heatmap, _ , count = find_cars(img, track.sw[i].starty, track.sw[i].endy,
            track.sw[i].startx, track.sw[i].endx, track.sw[i].scale, svc, X_scaler,
            orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        heat_map += heatmap
        total += count
    heat_map = apply_threshold(heat_map, track.threshold)
    labels = label(heat_map)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    return draw_img, heat_map, total

def testZoomHeatmap(path):
    '''
    A visualization method to test the tracking of cars in a heatmap at different scales.
    '''
    example_images = glob.glob(path)
    out_images = []
    out_titles = []
    for img_src in example_images:
        t1 = time.time()
        img = mpimg.imread(img_src)
        out_img, heat_map , count = trackZoomHeatmap(img)
        out_images.append(out_img)
        out_images.append(heat_map)
        out_titles.append(img_src[-9:]+' BBox')
        out_titles.append(img_src[-9:]+' Heatmap')
        print(time.time()-t1, 'seconds to run', count,'windows')
    fig = plt.figure(figsize=(12,12))
    visualise_plot(fig, len(out_images)//2, 2, out_images, out_titles)
    plt.show()

def visualizeMultiscaleSearch(path):
    '''
    Method to visualize the multiscale search windows.
    '''
    overlap = 0.0
    t1 = time.time()
    img = mpimg.imread(path)
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    for i in range(track.scales):
        print(track.sw[i].starty,track.sw[i].endy)
        windows = slide_window(img, x_start_stop=(track.sw[i].startx,track.sw[i].endx), 
            y_start_stop=(track.sw[i].starty,track.sw[i].endy),
            xy_window=(64*track.sw[i].scale,64*track.sw[i].scale), xy_overlap=(overlap, overlap))
        draw_img = draw_boxes(draw_img, windows, color=(0,0,255), thick=6)
    print(time.time()-t1, 'Seconds to process one image searching', len(windows),'windows')
    fig = plt.figure()
    plt.imshow(draw_img)
    plt.show()

class SearchWindow:
    '''
    Class to hold the search window parameters
    '''
    def __init__(self, startx=None, endx=None, starty=None, endy=None, scale=1.0, name=''):
        self.startx = startx
        self.endx = endx
        self.starty = starty
        self.endy = endy
        self.scale = scale
        self.name = name

class Tracker:
    '''
    Class to hold the current tracking metrics.
    '''
    def __init__(self, threshold=30, n_average=5, select=(1,1,1,1,1)):
        self.heat_tracks = []
        self.sw = []
        self.threshold = threshold
        self.n_average = n_average
        self.n_img = 0
        self.select = select
        self.scales = 0
        self.counter = 0
        if self.select[0]:
            # startx endx starty endy scale
            self.sw.append(SearchWindow( 300, 980, 400, 496, 0.5, 'very small')) # 32
            self.scales += 1
        if self.select[1]:
            self.sw.append(SearchWindow(300, 980, 400, 496, 1.0, 'small')) # 64
            self.scales += 1
        if self.select[2]:
            self.sw.append(SearchWindow(0, 1280, 400, 544, 1.5, 'medium')) # 96
            self.scales += 1
        if self.select[3]:
            self.sw.append(SearchWindow(0, 1280, 400, 690, 2.0, 'large')) # 128
            self.scales += 1
        if self.select[4]:
            self.sw.append(SearchWindow(0, 1280, 400, 690, 3.0, 'very large')) # 192
            self.scales += 1

def trackZoomHeat(img):
    '''
    Mulitzoom search and return heatmap
    '''
    total = 0
    img1 = img.astype(np.float32)/255
    heat_map = np.zeros_like(img1[:,:,0])
    for i in range(track.scales):
        draw_img, heatmap, _ , count = find_cars(img, track.sw[i].starty, track.sw[i].endy, 
            track.sw[i].startx, track.sw[i].endx, track.sw[i].scale, svc, X_scaler, 
            orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        heat_map += heatmap
        total += count
    return heat_map, draw_img, total

def processLaneLines(image):
    '''
    Process function for detecting and drawing the lane lines.
    '''
    # Apply a distortion correction to raw images.
    udist_img = cv2.undistort(image, mtx, dist, None, mtx)

    # Use color transforms, gradients, etc., to create a thresholded binary image.
    bin_img = lf.thresholdImage(udist_img)

    # Apply a perspective transform to rectify binary image ("birds-eye view").
    bird_img = lf.getBirdsEyeView(bin_img, pvt)
    
    # Detect lane pixels and fit to find the lane boundary.
    left_fit, right_fit, left_fit_cr, right_fit_cr = lf.findLaneLines(bird_img, pvt)
    
    # Determine the curvature of the lane and vehicle position with respect to center.
    curvature = lf.measureCurvature(bird_img.shape[0], left_fit_cr, right_fit_cr, pvt)

    # Warp the detected lane boundaries back onto the original image.
    final_img = lf.drawLane(bird_img, udist_img, left_fit, right_fit, pvt)

    # Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    final_img = lf.addTextOverlay(final_img, curvature, left_fit, right_fit, pvt)
    return udist_img, final_img

def visualizeImageSequence():
    '''
    Method to visualize the heatmap history buffer in an image sequence
    '''
    # 6 frames and corresponding heat maps
    example_images = glob.glob('./test_images/track*.jpg')
    out_images = []
    out_titles = []
    for img_src in example_images:
        img = mpimg.imread(img_src)
        out_images.append(img)
        out_img, heat_map , count = trackZoomHeatmap(img)
        out_images.append(heat_map)
        out_titles.append(img_src[-10:]+' Original')
        out_titles.append(img_src[-10:]+' Heatmap')
    fig = plt.figure(figsize=(12,14))
    visualise_plot(fig, len(out_images)//2, 2, out_images, out_titles)
    plt.savefig('./output_images/pv_6heat.png')
    plt.show()

    heat_stack = []
    for img_src in example_images:
        img = mpimg.imread(img_src)
        out_img, heat_map , count = trackZoomHeatmap(img)
        print(heat_map[450,650])
        heat_stack.append(heat_map)
    # Sum detections
    heat_tracks = np.asarray(heat_stack)
    heat_sum = np.sum(heat_tracks,0)
    print(heat_sum[450,650])
    plt.figure()
    plt.title('Summed Heatmap')
    plt.imshow(heat_sum, cmap='bone')
    plt.savefig('./output_images/pv_sum.png')
    plt.show()

    # Label cars
    heat_thresh = apply_threshold(heat_sum, 30)
    labels = label(heat_thresh)
    # Overlay bboxes
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    plt.figure()
    plt.title('Final Labeled Image')
    plt.imshow(draw_img)
    plt.savefig('./output_images/pv_final.png')
    plt.show()


def processHistory(img):
    '''
    The entry point to the image processing pipline. 
    Tracks cars using the HOG features and a Linear SVM.
    Employs a history buffer to filter spurious detections.
    The cars bounding boxes are drawn on the returned image.
    '''
    udist_img, final_img = processLaneLines(img)
    # Increment image counter
    track.n_img += 1

    # Generate multi-scale heat map for this image
    heat_map, bbox_img, _ = trackZoomHeat(udist_img)

    # The FIFO queue
    if track.n_img > track.n_average:
        track.heat_tracks.pop(0)
    track.heat_tracks.append(heat_map)
    
    # Sum detections
    heat_tracks = np.asarray(track.heat_tracks)
    heat_sum = np.sum(heat_tracks,0)
    
    # Label cars
    heat_thresh = apply_threshold(heat_sum, track.threshold)
    labels = label(heat_thresh)

    # Overlay bboxes
    draw_img = draw_labeled_bboxes(np.copy(final_img), labels)

    return draw_img #heat_sum.astype(np.float32)

# Main

# define global parameters
colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Number of binning dimensions
hist_bins = 32 # Number of histogram bins
spatial_feat = True # Spatial feature on or off
hist_feat = True # Historam features on or off
hog_feat = True # HOG features on or off

# get training set
cars, notcars = get_training_set()

#visualise_hog(colorspace=colorspace)
#visualise_spacial_binning(color_space=colorspace)

svc, X_scaler = loadLinearClassifier("svm_linear.p")
if svc is 0:
    svc, X_scaler = trainLinearClassifier()
    saveLinearClassifier("svm_linear.p", svc, X_scaler)

#trackImages('test_images/test8.jpg')
#trackCars('test_images/test*.jpg')

track = Tracker(threshold=5, n_average=0, select=(1,1,1,1,1))
#testZoomHeatmap('test_images/test*.jpg')
#visualizeMultiscaleSearch('test_images/test7.jpg')
#visualizeImageSequence()

# Process the videos
from moviepy.editor import VideoFileClip
from IPython.display import HTML

#lane finder
# source perspective points hand picked from an undistorted 'test_images/straight_lines2.jpg'
src_pts = np.float32(
        [[278,670],  # bottom left
         [1019,670], # bottom right
         [680,444],  # top right
         [603,444]]) # top left
# world height of src_pts from counting dashes and applying 3 meters per dash according to
# US regulations
world_ym = 33
# Create global perspective object
pvt = lf.Perspective(src_pts, world_ym, scale=1.0)
# Get camera matrix and distortion coeffecients (only do this once)
mtx, dist = lf.loadDistortion("dist_pickle.p")

if 0:
    t = 75
    n = 25
    track = Tracker(threshold=t, n_average=n, select=(1,1,1,1,1))
    project_output = 'output_videos/test_video_n{}_thresh{}_scale{}{}{}{}{}.mp4'.format(
        track.n_average, track.threshold, track.select[0], track.select[1], 
        track.select[2], track.select[3], track.select[4])
    clip1 = VideoFileClip("test_video.mp4")
    project_clip = clip1.fl_image(processHistory)
    project_clip.write_videofile(project_output, audio=False)

if 1:
    t = 90 #80 #110 #80
    n = 20 #15 #15 #25
    cstart = 38 #20 #10, 22, 38
    cend = 44 #23 #27, 27, 44
    track = Tracker(threshold=t, n_average=n, select=(1,1,1,1,1))
    # project_output = 'output_videos/project_video_{}to{}_n{}_thresh{}_scale{}{}{}{}{}.mp4'.format(
    #     cstart, cend,
    #     track.n_average, track.threshold, track.select[0], track.select[1], 
    #     track.select[2], track.select[3], track.select[4])
    project_output = 'output_videos/project_video.mp4'
    clip1 = VideoFileClip("project_video.mp4")
#    clip1 = clip1.subclip(cstart,cend)
    project_clip = clip1.fl_image(processHistory)
    project_clip.write_videofile(project_output, audio=False)
