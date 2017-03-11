import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.image as mpimg

# REMEMBER: when merging features, all features should be in order.


class FeatureExtractor(object):
    # Parameters to tweak

    # HOG Parameters
    orient = 9
    pix_per_cell = (8, 8)
    cell_per_block = (2, 2)
    hog_channel = 0 # Can be 0, 1 ,2 or 'ALL'
    transform_sqrt = False

    # Color Histogram Parameters:
    nbins = 32
    bins_range = (0, 256)

    # Spatial Bin Parameters:
    spatial_size = (16, 16)

    def __init__(self, lst_of_imgs=None, color_space='RGB', color_feature=True, hog_feature=True, bin_feature=True):

        self.images = lst_of_imgs
        self.color_space = color_space

        # Combined Feature
        self.features = []
        # Which features will be used
        self.enable_color = color_feature
        self.enable_hog = hog_feature
        self.enable_bin = bin_feature

        # HOG Feature
        self.hog_features = []
        self.hog_img = None

        # Color Histogram Feature
        self.color_features = []
        self.color_img = []

    def extract_feature(self, image):
        feature = []

        if self.enable_bin is True:
            bin_feat = self.bin_spatial(image)
            feature.append(bin_feat)

        if self.enable_color is True:
            _, color_feat = self.color_hist(image)
            self.color_features.append(color_feat)
            feature.append(color_feat)

        if self.enable_hog is True:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            hog_feat, hog_img = self.get_hog_features(gray)
            self.hog_features.append(hog_feat)
            feature.append(hog_feat)

        return feature

    def get_hog_features(self, img):
            hog_features, hog_img = hog(img, orientations=self.orient,
                                        pixels_per_cell=self.pix_per_cell,
                                        cells_per_block=self.cell_per_block,
                                        transform_sqrt=self.transform_sqrt,
                                        visualise=True, feature_vector=True)
            return hog_features, hog_img

    def color_hist(self, img):
        # Compute the histogram of the RGB channels separately
        # Concatenate the histograms into a single feature vector
        # Return the feature vector
        # Take histograms in R, G, and B
        rhist = np.histogram(img[:, :, 0], bins=self.nbins, range=self.bins_range)
        ghist = np.histogram(img[:, :, 1], bins=self.nbins, range=self.bins_range)
        bhist = np.histogram(img[:, :, 2], bins=self.nbins, range=self.bins_range)

        # Generating bin centers
        bin_edges = rhist[1]
        bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2

        # Histogram Features
        hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))

        return bin_centers, hist_features

    def bin_spatial(self, img):
        feature_img = None
        if self.color_space != 'RGB':
            if self.color_space is 'HSV':
                feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif self.color_space is 'HLS':
                feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif self.color_space is 'LUV':
                feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif self.color_space is 'YUV':
                feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif self.color_space is 'YCrCb':
                feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
        else:
            feature_img = np.copy(img)
        # Use cv2.resize().ravel() to create feature vector
        features = cv2.resize(feature_img, self.spatial_size).ravel()
        return features

    def get_feature(self):
        import time
        avg = []
        for img in self.images:
            t = time.time()
            image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
            feature = self.extract_feature(image)
            self.features.append(np.concatenate(feature))
            avg.append((time.time()-t))
        print("Average time / feature : {} seconds".format(np.average(avg)))
        return np.array(self.features)


