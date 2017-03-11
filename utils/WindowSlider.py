import numpy as np


class WindowSlider(object):
    def __init__(self, img, classifier, x_region=[None, None], y_region=[None, None], xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        self.windows = self.create_window(x_region, y_region, xy_window, xy_overlap)
        self.clf = classifier

    def create_window(self, x_region=[None, None], y_region=[None, None],
                      xy_window=(64, 64), xy_overlap=(0.5, 0.5)):

        if x_region[0] is None:
            x_region[0] = 0
        if x_region[1] is None:
            x_region[1] = 0
        if y_region[0] is None:
            y_region[0] = 0
        if y_region[1] is None:
            y_region[1] = 0
        # Compute the region of interest
        x = x_region[1] - x_region[0]
        y = y_region[1] - y_region[0]

        # Step size
        x_pix_per_step = np.int(xy_window[0]*(1-xy_overlap[0]))
        y_pix_per_step = np.int(xy_window[1]*(1-xy_overlap[1]))

        # Total windows
        x_windows = np.int(x/x_pix_per_step) - 1
        y_windows = np.int(y/y_pix_per_step) - 1

        windows = []
        for ys in range(y_windows):
            for xs in range(x_windows):
                # Calculate window pos
                startx = xs*x_pix_per_step + x_region[0]
                endx = startx + xy_window[0]
                starty = ys*y_pix_per_step + y_region[0]
                endy   = starty + xy_window[1]

                # Append window to list
                windows.append(((startx, starty), (endx, endy)))

        return windows
