import cv2
import time
import numpy as np
from skimage.feature import hog
import multiprocessing as mp


# ############################
# CONFIGURATION
# ############################

# HOG Parameters
orient = 9
pix_per_cell = (8, 8)
cell_per_block = (2, 2)
hog_channel = 'ALL'         # Can be 0, 1 ,2 or 'ALL'
transform_sqrt = False

# Color Histogram Parameters:
nbins = 32
bin_range = (0.0, 1.)
# Spatial Bin Parameters:
spatial_size = (32, 32)
color_space = 'YCrCb'


def get_feature(images, workers=4):
    pool = mp.Pool(processes=workers)
    avg, features = zip(*pool.map(process_img, images))
    print("Average time / feature : {} seconds".format(np.average(avg)))
    test = cv2.cvtColor(cv2.imread(images[0]), cv2.COLOR_BGR2RGB)/255
    print("Max Value {} Min Value {}\n".format(np.max(test), np.min(test)))
    return features


def process_img(img):
    t = time.time()
    image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
    feature = np.concatenate(extract_feature(image))
    avg = (time.time() - t)
    return avg, feature


def extract_feature(img):
    feature_img = None
    if color_space != 'RGB':
        if color_space is 'HSV':
            feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space is 'HLS':
            feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space is 'LUV':
            feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space is 'YUV':
            feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space is 'YCrCb':
            feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_img = np.copy(img)

    if (255 - np.max(feature_img)) > 100:
        feature_img = feature_img/255

    feature = []

    bin_feat = bin_spatial(feature_img)
    feature.append(bin_feat)

    color_feat = color_hist(feature_img)
    feature.append(color_feat)

    hog_feat = get_hog_features(feature_img, visual=False)
    feature.append(hog_feat)

    return feature


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


def get_hog_features(img, orient=9, pix_per_cell=8, cell_per_block=2,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32):  # bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features
