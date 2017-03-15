import cv2
import time
import numpy as np
from skimage.feature import hog
import multiprocessing as mp
from  threading import RLock, Lock

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

clahe = cv2.createCLAHE(clipLimit=2.0)


def get_feature(images, workers=4):
    pool = mp.Pool(processes=workers)
    results = []
    t = time.time()

    for img in images:
        image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        image = adaptive_equalize_image(image)
        results.append(image)

    avg, features = zip(*pool.map(process_img, results))
    print("Total time: {} seconds".format(time.time() - t))
    print("Average time / feature : {} seconds".format(np.average(avg)))
    test = cv2.cvtColor(cv2.imread(images[0]), cv2.COLOR_BGR2RGB)/255
    print("Max Value {} Min Value {}\n".format(np.max(test), np.min(test)))

    return features


def process_img(img):
    t = time.time()
    feature = np.concatenate(extract_feature(img))
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

    hog_feat = get_hog_features(feature_img)
    feature.append(hog_feat)

    return feature


def adaptive_equalize_image(img, level=2.0):
    """
    Equalize an image - Increase contrast for the image
        # http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html

    :param img:    an image
    :param level:  clipLevel
    :return: a equalized image
    """

    if img.shape[2] == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        result = cv2.merge((cl, a, b))
        result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
    else:
        result = clahe.apply(img)
    return result


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


def get_hog_features(img, ch='ALL', orient=9, pix_per_cell=8, cell_per_block=2, vis=False, feature_vec=True):
    if ch == 'ALL':
        hog_features = []
        for c in range(img.shape[2]):
            if vis is True:
                hog_feature, hog_img = hog(img[:, :, c], orientations=orient,
                                           pixels_per_cell=(pix_per_cell, pix_per_cell),
                                           cells_per_block=(cell_per_block, cell_per_block),
                                           transform_sqrt=transform_sqrt,
                                           visualise=True, feature_vector=feature_vec)
            else:
                hog_feature = hog(img[:, :, c], orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=transform_sqrt,
                                  visualise=False, feature_vector=feature_vec)

            hog_features.append(hog_feature)

        hog_features = np.ravel(hog_features)
    else:
        if vis is True:
            hog_features, hog_img = hog(img[:, :, ch], orientations=orient,
                                        pixels_per_cell=(pix_per_cell, pix_per_cell),
                                        cells_per_block=(cell_per_block, cell_per_block),
                                        transform_sqrt=transform_sqrt,
                                        visualise=True, feature_vector=feature_vec)
        else:
            hog_features = hog(img[:, :, ch], orientations=orient,
                               pixels_per_cell=(pix_per_cell, pix_per_cell),
                               cells_per_block=(cell_per_block, cell_per_block),
                               transform_sqrt=transform_sqrt,
                               visualise=False, feature_vector=feature_vec)
    if vis is True:
        return hog_features, hog_img
    else:
        return hog_features


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
