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


def color_hist(img):
    # Compute the histogram of the RGB channels separately
    # Concatenate the histograms into a single feature vector
    # Return the feature vector
    # Take histograms in R, G, and B
    rhist = np.histogram(img[:, :, 0], bins=nbins, range=bin_range)
    ghist = np.histogram(img[:, :, 1], bins=nbins, range=bin_range)
    bhist = np.histogram(img[:, :, 2], bins=nbins, range=bin_range)

    # Histogram Features
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    return hist_features


def bin_spatial(img):
    # Use cv2.resize().ravel() to create feature vector
    features = cv2.resize(img, spatial_size).ravel()
    return features


def get_hog_features(img, visual=False):
    if hog_channel == 'ALL':
        hog_features = []
        for c in range(img.shape[2]):
            if visual is True:
                hog_feature, hog_img = hog(img[:, :, c], orientations=orient,
                                           pixels_per_cell=pix_per_cell,
                                           cells_per_block=cell_per_block,
                                           transform_sqrt=transform_sqrt,
                                           visualise=True, feature_vector=True)
            else:
                hog_feature = hog(img[:, :, c], orientations=orient,
                                  pixels_per_cell=pix_per_cell,
                                  cells_per_block=cell_per_block,
                                  transform_sqrt=transform_sqrt,
                                  visualise=False, feature_vector=True)

            hog_features.append(hog_feature)

        hog_features = np.ravel(hog_features)
    else:
        if visual is True:
            hog_features, hog_img = hog(img[:, :, hog_channel], orientations=orient,
                                        pixels_per_cell=pix_per_cell,
                                        cells_per_block=cell_per_block,
                                        transform_sqrt=transform_sqrt,
                                        visualise=True, feature_vector=True)
        else:
            hog_features = hog(img[:, :, hog_channel], orientations=orient,
                               pixels_per_cell=pix_per_cell,
                               cells_per_block=cell_per_block,
                               transform_sqrt=transform_sqrt,
                               visualise=False, feature_vector=True)
    if visual is True:
        return hog_features, hog_img
    else:
        return hog_features

