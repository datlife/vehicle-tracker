import cv2
import time
import numpy as np
from skimage.feature import hog
import multiprocessing as mp
# Multi-threading:
# http://stackoverflow.com/questions/1816958/cant-pickle-type-instancemethod-when-using-pythons-multiprocessing-pool-ma?noredirect=1&lq=1

# HOG Parameters
orient = 9
pix_per_cell = (8, 8)
cell_per_block = (2, 2)
hog_channel = 0         # Can be 0, 1 ,2 or 'ALL'
transform_sqrt = False

# Color Histogram Parameters:
nbins = 32
bins_range = (0, 256)

# Spatial Bin Parameters:
spatial_size = (32, 32)
color_space = 'YUV'


def get_feature(images, workers=4):
    pool = mp.Pool(processes=workers)
    avg, features = zip(*pool.map(process_img, images))
    print("Average time / feature : {} seconds".format(np.average(avg)))
    return np.array(features)


def process_img(img):
    t = time.time()
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    feature = np.concatenate(extract_feature(image))
    avg = (time.time() - t)
    return avg, feature


def extract_feature(image):
    feature = []

    bin_feat = bin_spatial(image)
    feature.append(bin_feat)

    _, color_feat = color_hist(image)
    feature.append(color_feat)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hog_feat, hog_img = get_hog_features(gray)
    feature.append(hog_feat)

    return feature


def get_hog_features(img):
    hog_features, hog_img = hog(img, orientations=orient,
                                pixels_per_cell=pix_per_cell,
                                cells_per_block=cell_per_block,
                                transform_sqrt=transform_sqrt,
                                visualise=True, feature_vector=True)
    return hog_features, hog_img


def color_hist(img):
    # Compute the histogram of the RGB channels separately
    # Concatenate the histograms into a single feature vector
    # Return the feature vector
    # Take histograms in R, G, and B
    rhist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)

    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2

    # Histogram Features
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))

    return bin_centers, hist_features


def bin_spatial(img):
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
            feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    else:
        feature_img = np.copy(img)
    # Use cv2.resize().ravel() to create feature vector
    features = cv2.resize(feature_img, spatial_size).ravel()
    return features
