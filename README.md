# Vehicle Detection and Tracking
--------------------------------

This repo is the Project 5 of Udacity Self-Driving Car Nanodegree. 

In the next following sections, we would walk through the pipeline how to detect and track cars from a video stream using Support Vector Machine (SVM) Classifier. This process involves few intersting Computer Vision topics such as Histogram of Gradients (HOG), Spatial binary and Color Histogram.

We divided this tutorial the into several sections : 
1. [Extract Image Feature Vector](#1-extract-image-feature-vector)
2. [Train SVM Classifier](#2-train-svm-classifier) 
3. [Vehicle Detection](#3-vehicle-detection) 
4. [Vehicle Tracking](#4-vehicle-tracking)
5. [Video Pipeline](#5-video-pipeline)

## 1. Extract Image Feature Vector
The goal is to extract useful information from image so that the computer can quickly detect where is the car and where is not the car. One powerful way is to extract the Histogram of Gradients (Shape of the object), Color histogram (color of the object) and Spatial binary (overal feature of the object). Here is the example of a car and not-a-car image:
![](./docs/car-not-car.png)

#### Trick 1: Adaptive Histogram Equalization before extracting feature.
We discovered that the training data is somewhat blurry and noisy. In this project, we combine the Feature vector of each image so it is important to have a clear image for training. In Deep Learning approach, however, it might help the model generalize better.

By applying `Adaptive Histogram Equalization` (AHE), we could achieve better image quality. **The trade-off is speed**. Thus, we only apply AHE during training.**
```
def adaptive_equalize_image(img, level):
    """
    Adaptive Histogram Equalization : http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
    """
    clahe = cv2.createCLAHE(clipLimit=level, tileGridSize=(8, 8))
    if img.shape[2] == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        result = cv2.merge((cl, a, b))
        result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB) 
    else:
        result = clahe.apply(img)
    return result
```
![alt-text](./docs/adaptive.png)

#### Pre-processing:
`adaptive_histogram_equalization`: **True**

#### HOG Parameters


| Parameters      | Value |
|---------------- |-------| 
| `orientation`   | 9     |
| `pix_per_cell`  | 8     | 
| `cell_per_block`| 2     | 
| `hog_channel`   | `'ALL`'|
| `transform_srt` | False |
| `feature_vector`| True  |


#### Color Histogram Parameters:

| Parameters   | Value     |
|------------- |-----------| 
| `nbins`      | 32        |
| `bin_range`  | (0.0, 1.0)| 


#### Spatial Bin Parameters:
| Parameters   | Value   |
|------------- |---------| 
| `color_space`| `YCrCb` |
| `spatial_bin`| (32, 32)| 

In following graph, we extract features from two images and display in order of : Spatial Bin, Color Histogram, Histogram of Gradient
![](./docs/vector.png)

#### Normalize Feature Vector
To avoid bias over one feature, we need to normalize the data using `StandardScaler()`
```
from sklearn.preprocessing import StandardScaler
```
**Before normalization**
![](./docs/unorm-vector.png)

**After normalization**
![](./docs/normalized.png)

## 2. Train SVM Classifier
Next is to build a Support Vector Machine Classisifer to classify the image whether it is a car or not. Basically, SVM will convert the current dataset into higher dimentional space in order to make the classification process easier.
![](./docs/svm.png)

```
# svc = Pipeline([('scaling', StandardScaler()), ('classification', LinearSVC(loss='hinge')),])
svc = SupportVectorMachineClassifier()
# Apply Standard Scalars to normalize vector
svc.train(x_train, y_train)
# Test on testing set
score = svc.score(x_test, y_test)

OUTPUT:
Starting to train vehicle detection classifier.
Completed training in 4.649303 seconds.

Testing accuracy:
Accuracy 0.992399%
```

## 3. Vehicle Detection
There will be two parts:

* Using sliding technique: slow and provide a lot of False Positives
* Using subsampling HOG and adding heatmap threshold: faster and eliminate a lot of False Positives

1. Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?
I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;)

2. Show some examples of test images to demonstrate how your pipeline is working. What did you do to optimize the performance of your classifier?

## 4. Vehicle Tracking

We created two object in order to make the tracking task easier: `Vehicle` and `VehicleTracker`. Both are available under `utils/VehicleTracker.py`

Blob and Watershed algorithms are suitable in this problem. I tried to implemented from stratch, however, it is good to know that OPENCV provided the algorithm so no need to "reinvent-the-wheels" unless you want to learn.


* `Vehicle` object holds information about the bounding box, pixels belong to an vehicle and old bounding boxes.
* `VehicleTracker` object keeps track a list of current tracked vehicles and making new adjustments based on new heatmaps from video stream.
```
class Vehicle(object):
    # A flavor of Blob algorithm
    def __init__(self, x, y):
        self.id = 1
        # Bounding Box
        self.minx = x
        self.miny = y
        self.maxx = x
        self.maxy = y
        # Collection of pixels in that bounding box
        self.points = []
        self.old_bbox = []
        self.points.append((x, y))
        
    def add(self, x, y):
        # Add a new pixel into current Vehicle
        self.points.append((x, y))  
        # Update the bounding box
        # Get the current min and max pixel - to avoid left box stay stilled.
        self.minx = min(self.minx, x)
        self.miny = min(self.miny, y)
        self.maxx = max(self.maxx, x)
        self.maxy = max(self.maxy, y)

    def isNear(self, x, y):
        # Determine if a new pixel (x, y) is belong to current object 
        # Get The center of the box
        cx = (self.minx + self.maxx)/2
        cy = (self.miny + self.maxy)/2
        if cx == x and cy == y:
            return True
        # Distance(edge, pixel)
        dist = self.distSqt(cx, cy, x, y)

        # If distance is under a cetrain threshold. we can determine this pixel belongs to the current object
        # If blob is too small, update this10Vehicle.distance_threshold:
        if dist < 100:
            return True
        else:
            return False

    def distSqt(self, x1, y1, x2, y2):
        # Calculate Distance between two pixels
        delta_x = x1-x2
        delta_y = y1-y2
        dist = np.sqrt(delta_x**2 + delta_y**2)
        return dist

    def clear(self):
        ''' Clear hot pixels from previous frames, except the boundary'''
        # Save old bounding box
        self.old_bbox.append(((self.minx, self.miny), (self.maxx, self.maxy)))
        if len(self.old_bbox) > 10:
            self.old_bbox.pop(0)
        self.minx = 1280
        self.miny = 720
        self.maxx = 0
        self.maxy = 0
        self.points = []
        
    def get_center(self):
        # Get The edge of bounding box that closest to (x,y)
        cx = (self.minx + self.maxx)/2
        cy = (self.miny + self.maxy)/2
        return (cx, cy)
    
    def size(self):
        return abs(self.maxx - self.minx)*abs(self.maxy - self.miny)
    
    def show(self):
        if len(self.old_bbox) > 2:
            self.update_box(lr=0.5)
        return (self.minx, self.miny), (self.maxx, self.maxy)
    
    def in_frame(self):
        area = abs(self.maxx - self.minx)*abs(self.maxy - self.miny)
        if area is 0 : return False
        if len(self.points)/area < 0.3:
            return False
        else:
            return True
    
    def get_poins(self):
        return self.points
    
    def update_box(self, lr=0.6):
        # Get the current min and max pixel
        (minx, miny), (maxx, maxy) = np.average(self.old_bbox, axis=0).astype(np.int)

        self.minx = int((1-lr)*self.minx + lr*minx)
        self.miny = int((1-lr)*self.miny + lr*miny)
        self.maxx = int((1-lr)*self.maxx + lr*maxx)
        self.maxy = int((1-lr)*self.maxy + lr*maxy)
```

```
class VehicleTracker(object):
    
    def __init__(self, looking_back_frames=10):
        #  List of previous heat_maps
        self.heat_maps = []
        # How far to look back
        self.smooth_factor = looking_back_frames
        #  Tracked Vehicles
        self.tracked_vehicles = []

    def update(self, new_heat_map, heat_threshold=10):
        
        # If we are just started to recording
        if len(self.heat_maps) < self.smooth_factor:
            self.heat_maps.append(new_heat_map)
            updated_map = np.sum(self.heat_maps, axis=0)
            # Remove objects that are not cars - low threshold
            updated_map[updated_map <= (heat_threshold*(len(self.heat_maps)*1.5/self.smooth_factor))] = 0
        else:
            # Add new map to current heatmap
            self.heat_maps.append(new_heat_map)
            updated_map = np.sum(self.heat_maps, axis=0)
            
            # Remove the earliest heat map
            earliest_map = self.heat_maps.pop(0)
            updated_map -= earliest_map
            
            # Remove objects that are not cars - low threshold
            updated_map[updated_map <= heat_threshold] = 0

        # list of all detected cars
        potential_cars = self.find_vehicles(updated_map)
        
        # Check if car is far from us
        self.tracked_vehicles = [car for car in potential_cars if (car.size() > 2200) is True]
        
        # Generate bounding boxes
        bboxes = [(car.show()) for car in self.tracked_vehicles]
        
        return bboxes, updated_map
    
    def find_vehicles(self, updated_heatmap, threshold=3.):   
        
        # Find pixels with value greater than threshold
        x, y = (updated_heatmap > threshold).nonzero()
        hot_pixels = np.dstack((y, x))[0].tolist()
        
        # Clear points from previous frames excepts the boundary
        for i in self.tracked_vehicles: i.clear()
   
        # Merge a group of pixels into one car if their distance are close to each other
        for pixel in hot_pixels:
            if len(self.tracked_vehicles) is 0:
                self.tracked_vehicles.append(Vehicle(pixel[0],pixel[1]))       
            else:
                found = False
                for vehicle in self.tracked_vehicles:
                    if vehicle.isNear(pixel[0], pixel[1]) is True:
                        vehicle.add(pixel[0], pixel[1]) # add and update bounding box
                        found = True
                        break               
                if found is False:
                    # Can I combine?                        
                    self.tracked_vehicles.append(Vehicle(pixel[0], pixel[1])) 
        
        # If car is not in a new frame, then remove
        self.tracked_vehicles = [car for car in self.tracked_vehicles if car.in_frame()]
        
        # Check intersect
        # http://stackoverflow.com/questions/642763/find-intersection-of-two-lists
        return self.tracked_vehicles
```


## 5. Video Implementation

1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.) 

Here's a link to my video result: [result](./result.mp4)

2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used scipy.ndimage.measurements.label() to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.

```
# Parameter
xstart = [500, 0]
xstop =  [1280, 1280]
ystart = [400, 400]
ystop =  [506,  656]
scales = [0.75,1.35]
# Vehicle Tracker
car_tracker = VehicleTracker(looking_back_frames=15)

def process_image(frame):
    global svc
    global car_tracker
    global ystar, ystop, scale
    svc_img = np.copy(frame)
    
    heatmaps = []
    # Multi-scale window
    for i, scale in enumerate(scales):
        heatmap, windows = find_cars(frame,xstart[i], xstop[i], ystart[i], ystop[i], scale, svc, dec_thresh=0.99)
        heatmaps.append(heatmap)
        
    # Combine heat map
    heatmap = np.sum(heatmaps, axis=0)
    heatmap[heatmap <= 5.] = 0
    cars, heatmap = car_tracker.update(heatmap, heat_threshold=30)
    
    # Draw car boxes
    for p1, p2 in itertools.chain(cars):
        cv2.rectangle(svc_img, p1, p2, (255, 255, 255), 4)
    # Create an heat image
    img = 255*heatmap/np.max(heatmap)
    img = np.dstack((img, heatmap, heatmap)).astype(np.uint8)
    svc_img = cv2.addWeighted(svc_img, 0.8, img, 1.0, 0.0)
    return svc_img
```

## Discussion
1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

The pipeline still detects a few false positives. It is critical to NOT HAVE False positives. There are a few advanced OPENCV topics could improve the result of the project such as image segmentation to avoid overlapped bounding boxes.

If I have more time, I would rather use CNN combined with Image Sengmentation Method to improve the tracking process.
