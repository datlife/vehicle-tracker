import os, fnmatch
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def data_look(car_list, not_car_list):
    # Define a function to return some characteristics of the data set

    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    # Define a key "n_notcars" and store the number of notcar images
    data_dict['n_notcars'] = len(not_car_list)
    data_dict['n_cars'] = len(car_list)

    # Read in a test image, either car or notcar
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

    print('Your function returned a count of',
          data_info["n_cars"], ' cars and',
          data_info["n_notcars"], ' non-cars')
    print('of size: ', data_info["image_shape"], ' and data type:',
          data_info["data_type"])
    # Just for fun choose random car / not-car indices and plot example images
    car_ind = np.random.randint(0, len(cars))
    notcar_ind = np.random.randint(0, len(not_cars))

    # Read in car / not-car images
    car_image = mpimg.imread(cars[car_ind])
    notcar_image = mpimg.imread(not_cars[notcar_ind])

    # Plot the examples
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(car_image)
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(notcar_image)
    plt.title('Example Not-car Image')
    plt.show()
