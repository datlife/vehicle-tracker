
import numpy as np
import cv2
import os, fnmatch
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
    # Return a list of filenames in a given folder with certain pattern
    # images = glob.glob('../data/**.jpeg')
    images = []
    for root, dirnames, filenames in os.walk(src_path):
        for filename in fnmatch.filter(filenames, pattern):
            images.append(os.path.join(root, filename))
    return images


def get_hog_features(img, orient=9, pix_per_cell=8, cell_per_block=(2, 2), vis=False, feature_vec=True, transform_sqrt=False):

    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt= transform_sqrt,
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt= transform_sqrt,
                       visualise=False, feature_vector=feature_vec)
        return features


if __name__ == "__main__":
    images = get_file_names('../test_images/', pattern='*.jpg')
    cars = []
    not_cars = []
    for img in images:
        if 'image' in img or 'extra' in img:
            not_cars.append(img)
        else:
            cars.append(img)

    data_info = data_look(cars, not_cars)

    # Generate a random index to look at a car image
    ind = np.random.randint(0, len(cars))
    # Read in the image
    image = mpimg.imread(cars[ind])
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Define HOG parameters
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    # Call our function with vis=True to see an image output
    features, hog_image = get_hog_features(gray, orient,
                                           pix_per_cell,
                                           cell_per_block,
                                           vis=True, feature_vec=False)

    # Plot the examples
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Visualization')
    plt.show()
