import cv2
import os, fnmatch
import numpy as np
from skimage.feature import hog
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def data_look(car_list, not_car_list):
    # Define a function to return some characteristics of the data set

    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    # Define a key "n_notcars" and store the number of notcar images
    data_dict['n_notcars'] = len(not_car_list)
    data_dict['n_cars'] = len(car_list)

    # Read in a test image,
    sample = cv2.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict['image_shape'] = np.shape(sample)
    # Define a key "data_type" and store the data type of the test image.
    data_dict['data_type'] = sample.dtype
    return data_dict


def get_file_names(src_path='./', pattern='*.jpeg'):
    # Return a list of file names in a given folder with certain pattern
    # images = glob.glob('../data/**.jpeg')
    images = []
    for root, dir_names, file_names in os.walk(src_path):
        for filename in fnmatch.filter(file_names, pattern):
            images.append(os.path.join(root, filename))
    return images


def get_hog_features(img, orient=9, pix_per_cell=8, cell_per_block=(2, 2),
                     vis=False, feature_vec=True, transform_sqrt=False):

    if vis:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=transform_sqrt,
                                  visualise=True,
                                  feature_vector=False)
        return features, hog_image
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=transform_sqrt,
                       visualise=False,
                       feature_vector=feature_vec)
        return features

from sklearn.preprocessing import StandardScaler


def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                     hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for img in imgs:
        # Read in each one by one
        img = mpimg.imread(img)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        else:
            feature_image = np.copy(img)
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)

        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)

        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features)))
    # Return list of feature vectors
    return features


def bin_spatial(img, size=(32, 32)):
    # Define a function to compute binned color features
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Define a function to compute color histogram features
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

if __name__ == "__main__":
    images = get_file_names('../data/', pattern='*.jpeg')
    cars = []
    not_cars = []
    for img in images:
        if 'image' in img or 'extra' in img:
            not_cars.append(img)
        else:
            cars.append(img)

    data_info = data_look(cars, not_cars)

    car_features = extract_features(cars, cspace='YUV', spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256))
    notcar_features = extract_features(not_cars, cspace='YUV', spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256))
    if len(car_features) > 0:
        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)

        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        car_ind = np.random.randint(0, len(cars))
        sample = mpimg.imread(cars[car_ind])
        gray = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
        _, car_histo = hog(gray, visualise=True)
        # Plot an example of raw and scaled features
        fig = plt.figure(figsize=(12, 4))
        plt.subplot(141)
        plt.imshow(mpimg.imread(cars[car_ind]))
        plt.title('Original Image')
        plt.subplot(142)
        plt.imshow(car_histo, cmap='gray')
        plt.title('HOG Image')
        plt.subplot(143)
        plt.plot(X[car_ind])
        plt.title('Raw Features')
        plt.subplot(144)
        plt.plot(scaled_X[car_ind])
        plt.title('Normalized Features')
        fig.tight_layout()
        plt.show()
    else:
        print('Your function only returns empty feature vectors...')