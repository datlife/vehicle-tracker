import cv2
import os, fnmatch
import numpy as np
import time
from utils import FeatureExtractor, WindowSlider
from utils import SVC
from utils import get_feature


def get_file_names(src_path='./', pattern='*.jpeg'):
    # Return a list of file names in a given folder with certain pattern
    # images = glob.glob('../data/**.jpeg')
    images = []
    for root, dir_names, file_names in os.walk(src_path):
        for filename in fnmatch.filter(file_names, pattern):
            images.append(os.path.join(root, filename))
    return images

# Import car and not car images
cars = get_file_names('./data/vehicles', pattern='*.png')
not_cars = get_file_names('./data/non-vehicles', pattern='*.png')

# Calculate car features & not-car features
t = time.time()
print("Calculating features for {} images...".format(len(cars)+len(not_cars)))
# I could not perform multi-processing on class object so,
# car_features = FeatureExtractor(cars, color_space='YUV').get_feature()
# I decided to use regular method
car_features = get_feature(cars, workers=4)
not_car_features = get_feature(not_cars, workers=4)
print("Completed calculating feature in {:f} seconds\n".format((time.time() - t), 3))

# Create data set
x = np.vstack((car_features, not_car_features)).astype(np.float64)
y = np.concatenate((np.ones(len(car_features)), np.zeros(len(not_car_features))))
print("Car Feature Vector's length: ", len(car_features))
print("Not Car Feature Vector's length: ", len(not_car_features))

# Training using SVC Classifier
svc = SVC(x, y, test_split=0.01)
svc.train()
svc.score()


# Create windows
window = WindowSlider()