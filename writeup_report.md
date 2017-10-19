## Vehicle Detection Project Writeup

---

### **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/visualise_RGB_HOG_features.png
[image2]: ./output_images/visualise_YCrCb_HOG_features.png
[image3]: ./output_images/visualise_RGB_hist_features.png
[image4]: ./output_images/visualise_YCrCb_hist_features.png
[image5]: ./output_images/visualise_RGB_spatial_features.png
[image6]: ./output_images/visualise_YCrCb_spatial_features.png
[image7]: ./output_images/windows_test.png
[image8]: ./output_images/visualize_multiscale.png
[image9]: ./output_images/visualize_pipeline.png
[image10]: ./output_images/pv_6heat.png
[image11]: ./output_images/pv_sum.png
[image12]: ./output_images/pv_final.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### In this writeup report I will consider the rubric points individually and describe how I addressed each point in my implementation.

---


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The HOG features were extracted from the training images in a function called `get_hog_features()` which was taken directly from the Udacity lesson and required no modifications. `Line XX`.

First, I implemented a function called `visualize_hog` to visualize the hog features with various color channels as well as various bins, orienations, and sizes. I then ran this function on several random images from both the `vehicle` and the `non-vehicle` classes with various channel settings and compared the results. The figure below illustraits the result of HOG detection in the `RGB` channel and HOG parameters of `orientations=6`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` on a `vehicle` and `non-vehicle` image:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

It is interesting to see that in the case of the `non-vehicle` HOG plot that there is more information in the Cr and Cb channels compared with the corresponding G abd B channels in the RGB plot.  This indicated that `YCrCb` would make a better feature for detecting `non-vehicles`. This effect was seen in the `vehicle` image as well.

Using the `bin_spacial()` and `color_hist()` functions, on lines `XX` and `XX`, I then extracted spatial binning and the color histogram features for both `RGB` and `YCrCb` to see what additional information they could provide:

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]

All plots show that using the spatial color binning and color histograms provide a unique signature for both `vehicle` and `non-vehicle`. Looking at the historgram information differences between `RGB` and `YCrCb` its apparent that the `YCrCb` colorspace has a very different response in the last tw channels.  What makes this good as a feature detector is that are not that similar to each other.  For example, in the last subplot in the image above, `Cr` seems to pull upward and `Cb` seems to pull downward. The dynamic range of `CrCb` is much less than its `GB` counterpart.

#### 2. Explain how you settled on your final choice of HOG parameters.

There were plenty of parameters availible to create a robust HOG featureset which made it tricky and time consuming to try each combination.  Having the `visualize_hog` certainly sped up the process and through many iterations I finally decided on the following parameters:

| Paramter | Value |
|:--------:|:-----:|
|colorspace|YCrCb|
|orient | 9|
|pix_per_cell| 8|
|cell_per_block |2|
|hog_channel| ALL|
|spatial_size| (32, 32)|
|hist_bins| 32|
|spatial_feat| True |
|hist_feat| True |
|hog_feat |True |

I knew the `colorspace` I wanted to use was one of the `non-RGB` spaces like  `LUV`,`YUV`, and `YCrCb` because they are commonly known to contain more distinct information over RGB and is more robust to shadows and illumination.  However the final choice of `YCrCb` over the other chroma/luma channels was decided on by best training results seen from experimentation in the Linear SVM classifier.

The number of `orientation` bins did not impact performance so it was deciced to use the optimal value from the original paper: http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf

The `pix_per_cell` and `cell_per_block` were chosen from the class lectures as they performed well on 64x64 pixel images. Larger `pix_per_cell` showed over generalized gradients where as higher values ran the risk of over fitting the training images.

Using all color channels for the HOG feature creates a bigger feature but performed much better at any one single channel in the linear SVM classifier.

The number of binning dimensions 

Adding the spacial and histogram features provided more useful information about the images to the be classified.  Based on the results seen in the above plots it made sense to add more unique information to the final feature descriptor to add robustness to the classifier.

The final feature vector length was `8460`.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the following code block on line `XX`:

```python
    # get training set
    cars, notcars = get_training_set()
    
    svc, X_scaler = loadLinearClassifier("svm_linear.p")
    if svc is 0:
        svc, X_scaler = trainLinearClassifier()
        saveLinearClassifier("svm_linear.p", svc, X_scaler)
```

It first calls the `get_training_set()` function on line `XX` to load up `GTI` `KITTI` and `Extras` data set as file names. The images are loaded and the SVM is trained in the `trainLinearClassifier()` function on line `XX`.

The training set consisted of 8792 `vehicle` and 8968 `not-vehicle` samples which were split into training and test sets where 10% was held back for testing.  The `sklearn.sm.LinearSVC` function with a linear kernel and default paramters were used to fit the data set.

It took 97 seconds to compute the feature aand 9.6 seconds to train the SVC and resulted in an final training set accuracy of 0.99.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I first used a window size of 96x96 pixels with an overlap of 50% on the entire image to see how it would perform in the `trackImages() function on line `XX`:

![alt text][image7]

The result of this is displayed below. The false positives in the trees and scenery prompted me to restricted the search to the bottom half of the screen. 

I tested various overlaps and window sizes to get a feel of how well it performed. 50% overlap was faster than 75% overlap, but the window bounaries on the cars were a bit coarse.  A finer windows size of 64x64 provided better results.

I created the function `visualizeMultiscaleSearch()` on line `XX` to visualize a multiscale approach to the sliding windows.  Multiscale windows facilitates detecting vehicles at different scales or sizes into the foreground of the horizon. The `Tracker` class on line `XX` is used to hold the search window parameters.  There are 5 scales availible to choose from with the following values:

|name|start_x|end_x|start_y|end_y|scale|size|
|:------:|:-----:|:------:|:-----:|:------:|:-----:|
|very small|300|980|400|496|0.5|32|
|small|300|980|400|496|1.0|64|
|medium|0|1280|400|544|1.5|96|
|large|0|1280|400|690|2.0|128|
|very large|0|1280|690|496|3.0|192|

Here is a visualization of the final search space:

![alt text][image8]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The next step was to successively add the detection windows into a heatmap. This heat map was thresholded and bounding boxes drawn around the remaining blobs. Here is an example of applying a heatmap to a set of detections and the final bounding box drawn in the original image:

To optimize the performance of the classifier I made sure that it has a test accuracy better than 98% before making any adjustments to the pipeline after this step. Ultimately I searched on five scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image9]

Once this was achieved it was still taking a long time to search for detections in each frame.  The best method to improve the query time was to detect the HOG features for the whole search area before sliding the detector over each search window.  This was done in the `find_cars()` function on line `XX`.  The bounding boxes were improved in fit and resolution by adding more overlap to the search but this also resulted in a slower response time.  The final window count of 1046 windows was a good compromize allowing the classifier to fullyclassify an image in just over 1 second.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_videos/project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. This all happens in the `processHistory()` function on line `XX`.

I added two filters to help with false positives. The first was an accumulator that added the last 15 images before thresholding was performed.  This removed the effect of spurious detections as they become less evident as the size of the history buffer increases.  This was implemented by a simple FIFO queue in the process loop on line `XX`.  It was necessary to increase the threshold to a much larger value of `80` after this method was added. The second filter was a size filter that restricted the minimum size of the drawn bounding boxes to no less than 50 pixels in width and height.  This was added to the `draw_labeled_bboxes()` function on line `XX`.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image10]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

![alt text][image11]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image12]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I approached the problem in a systematic way.  First I trained a linear SVM classifier using HOG features and optimized the parameters and feature descriptor to generate the best test result. I finally achieved close to 99% test accuracy.  After this I used the SVM to classify test images using a sliding window with various scales, overlap, and locations.  This worked fairly well but had issues with some objects like trees and shadows.  Fine tuning the size of the search windows annd restricting the search area to only the road ahead of the car helped a lot with this. Adding a heatmap, thresholding, and surrounding blobs with bounding boxes was easy to do given that the `sklearn` package provides a handy `Labels` function to segment the blobs.  I finally I wrapped it all into an image processing pipeline and discovered that there were some spurious false detections. To combat this I added a history buffer that summed the last 15 heatmaps which filtered out the spurious false positives quite well.  The length of the buffer and the threshold values were tuned from trial and error on smaller test video segments fromthe original video.

The result is a fairly good car tracker for these kind of conditions.  It is sensitive to shadows, some guard rails that look like a bit like car windows, and traffic approaching on the other side of the road, although this can be argued to be an actual true positive result. Adding the heatmap accumulator reduces the responsiveness of the algorithm because detection may be delayed up to 1/2 second.  This may not be good for any sense and avoid behaviours as it would also delay the response time to an event like a collision.  A big issue with the tracker is the speed of inferencing.  My implementation took just over one second per image frame to detect and segment cars on my machine.  This is far from real time and self driving car processors are generally not as powerful as high end desktop computer. 

The next step to improve this algorithm would be to speed it up by using an image pyramid.  One could perform classification on a lower resolution image to get interest area, then these could be used to seed search regions for a subsequent higher resolution pass. There are some deep learning models that may be faster was well, for example using [Yolo](https://pjreddie.com/darknet/yolo/) instead of the SVM and sliding window approach to get real time object detection. Another way to improve the algorithm is add velocity to the tracked objects.  Assigning velocity to an object makes it easier to predict where it is going to be next.  This can reinforce our detections in the next frame as well as help us recover from occlusion from things like another passing vehicle.


