import numpy as np
import cv2
from utils.feature_extractor import extract_feature

def slide_window(img_size, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """Returns all windows to search in an image.
    No classification has been done at this stage.
    """
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img_size[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] =  img_size[0]

    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))

    # Compute the number of windows in x/y
    nx_windows = np.int(xspan / nx_pix_per_step) - 1
    ny_windows = np.int(yspan / ny_pix_per_step) - 1

    # Initialize a list to append window positions to
    window_list = []

    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


def search_windows(frame, windows, clf, size=(64, 64), decision_threshold=0.3):
    on_windows = []
    for window in windows:
        # Get a region of an image
        region = cv2.resize(frame[window[0][1]:window[1][1], window[0][0]:window[1][0]], size)

        # Extract feature of mini image
        region_feature = np.concatenate(extract_feature(region))
        region_feature = region_feature.astype(np.float64).reshape(1, -1)

        # Pedict using your classifier
        dec = clf.decision_function(region_feature)
        prediction = int(dec > decision_threshold)

        # If positive (prediction == 1) then save the window
        if prediction == 1:
            # rint"Found it!"
            on_windows.append(window)
    return on_windows


class WindowSlider(object):
    def __init__(self, x_region=[None, None], y_region=[None, None], xy_window=(96, 96), xy_overlap=(0.5, 0.5), default_window_size=(64, 64)):
        self.windows = self.slide_window(x_region, y_region, xy_window, xy_overlap)
        self.window_size = default_window_size
        self.overlap = xy_overlap

    # Create a list of windows for SVC to select a region of image and compare with the feature vector
    def slide_window(self, img_size, x_start_stop=[None, None], y_start_stop=[None, None],
                     xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        """Returns all windows to search in an image.
        No classification has been done at this stage.
        """
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img_size[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img_size[0]

            # Compute the span of the region to be searched
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]

        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))

        # Compute the number of windows in x/y
        nx_windows = np.int(x_span / nx_pix_per_step) - 1
        ny_windows = np.int(y_span / ny_pix_per_step) - 1

        # Initialize a list to append window positions to
        window_list = []

        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                start_x = xs * nx_pix_per_step + x_start_stop[0]
                end_x = start_x + xy_window[0]
                start_y = ys * ny_pix_per_step + y_start_stop[0]
                end_y = start_y + xy_window[1]

                window_list.append(((start_x, start_y), (end_x, end_y)))
        # Return the list of windows
        return window_list

    def search_windows(self, frame, clf, decision_threshold=0.3):
        on_windows = []
        for window in self.windows:
            # Get a region of an image
            region = cv2.resize(frame[window[0][1]:window[1][1], window[0][0]:window[1][0]], self.window_size)

            # Extract feature of mini image
            region_feature = np.concatenate(extract_feature(region))
            region_feature = region_feature.astype(np.float64).reshape(1, -1)

            # Predict using your classifier
            dec = clf.decision_function(region_feature)
            prediction = int(dec > decision_threshold)

            # If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append(window)
        return on_windows



