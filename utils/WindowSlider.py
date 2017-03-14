import numpy as np
import cv2
from utils.feature_extractor import extract_feature, get_hog_features, bin_spatial, color_hist, convert_color

# Define a single function that can extract features using hog sub-sampling and make predictions


def find_cars(img, ystart, ystop, scale, svc, dec_thresh=0.75, orient=9, pix_per_cell=8, cell_per_block=2,
              spatial_size=32, hist_bins=32):
    img = img.astype(np.float32) / 255

    # Make a heat map of zero
    heatmap = np.zeros_like(img[:, :, 0])

    img_to_search = img[ystart:ystop, :, :]
    ctrans_to_search = convert_color(img_to_search, conv='RGB2YCrCb')

    if scale != 1:
        imshape = ctrans_to_search.shape
        ctrans_to_search = cv2.resize(ctrans_to_search, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    # Define blocks and steps as above
    nxblocks = (ctrans_to_search.shape[1] // pix_per_cell) - 1
    nyblocks = (ctrans_to_search.shape[0] // pix_per_cell) - 1
    nfeat_per_block = orient * cell_per_block ** 2
    # 64 was the original sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ctrans_to_search, ch=0, feature_vec=False)
    hog2 = get_hog_features(ctrans_to_search, ch=1, feature_vec=False)
    hog3 = get_hog_features(ctrans_to_search, ch=2, feature_vec=False)

    img_boxes = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_to_search[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=(spatial_size, spatial_size))
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
            # Pedict using your classifier
            dec = svc.decision_function(test_features)
            prediction = int(dec > dec_thresh)

            # If positive (prediction == 1) then save the window
            if prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                heatmap[ytop_draw + ystart:ytop_draw + win_draw + ystart, xbox_left:xbox_left + win_draw] += 1
                img_boxes.append(
                    ((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + ystart + win_draw)))

    return heatmap, img_boxes


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
            on_windows.append(window)
    return on_windows


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





