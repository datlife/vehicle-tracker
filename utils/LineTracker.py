import cv2
import numpy as np


class LineTracker(object):
    def __init__(self, window_width, window_height, margin, xm, ym, smooth_factor):
        self.window_width = window_width
        self.window_height = window_height
        self.margin = margin
        self.ym_per_pixel = ym
        self.xm_per_pixel = xm
        self.smooth_factor = smooth_factor
        self.recent_centers = []           # Store recent windows (left,right)

    def find_lane_line(self, warped):
        wd_w = self.window_width
        wd_h = self.window_height
        margin = self.margin
        r = warped.shape[0]
        c = warped.shape[1]
        wd_centroids = []   # store (left, right) windows centroid position per level
        windows = np.ones(wd_w)  # Window Template for convolution

        # First find the two staring positions for the left and right lane using np.sum
        # to get the vertical image slice.

        # Then, use np.convolve the vertical image slice with the window template

        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(warped[int(3*r/4):, :int(c/2)], axis=0)
        l_center = np.argmax(np.convolve(windows, l_sum)) - wd_w/2
        r_sum = np.sum(warped[int(3*r/4):, int(c/2):], axis=0)
        r_center = np.argmax(np.convolve(windows, r_sum)) - wd_w/2 + int(c/2)

        # Add what we found for the first layer
        wd_centroids.append((l_center, r_center))

        # Iterate through each layer looking for max pixel locations
        for level in range(1, int(r/wd_h)):
            # convolve the window into the vertical slice of the image
            img_layer = np.sum(warped[int(r - (level+1)*wd_h):int(r-level*wd_h), :], axis=0)
            conv_signal = np.convolve(windows, img_layer)
            # Find the est left centroid using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at the right side of window,
            # not the center of the window
            offset = wd_w/2
            l_min_index = int(max(l_center+offset-margin, 0))
            l_max_index = int(min(l_center+offset+margin, c))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset

            # Find the  best right centroid
            r_min_index = int(max(r_center+offset-margin, 0))
            r_max_index = int(min(r_center+offset+margin, c))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset

            # Update window_centroids
            wd_centroids.append((l_center, r_center))

        self.recent_centers.append(wd_centroids)

        # Smooth the line
        return np.average(self.recent_centers[-self.smooth_factor:], axis=0)

    def curve_fit(self, img, leftx, rightx):
        '''
        Find a polynomial function of left and right lanes as:
        F(x) = Ax^2 + Bx + C
        '''

        res_yvals = np.arange(img.shape[0] - (self.window_height/2), 0, -self.window_height)
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        y_idx = np.int32(ploty)

        # Fit into a function - return coefficients
        # Left function
        l_coeffs = np.polyfit(res_yvals, leftx, deg=2)
        left_fitx = l_coeffs[0]*ploty**2 + l_coeffs[1]*ploty + l_coeffs[2]
        left_fitx = np.int32(left_fitx)[0:img.shape[0]]

        # Right function
        r_coeffs = np.polyfit(res_yvals, rightx, deg=2)
        right_fitx = r_coeffs[0]*ploty**2 + r_coeffs[1]*ploty + r_coeffs[2]
        right_fitx = np.int32(right_fitx)[0:img.shape[0]]

        left_lane = np.array(list(zip(np.concatenate((left_fitx - self.window_width/2,
                                                      left_fitx[::-1]+self.window_width/2), axis=0),
                                      np.concatenate((ploty, ploty[::-1]), axis=0))), dtype='int32')
        right_lane = np.array(list(zip(np.concatenate((right_fitx - self.window_width/2,
                                                      right_fitx[::-1]+self.window_width/2), axis=0),
                                       np.concatenate((ploty, ploty[::-1]), axis=0))), dtype='int32')
        inner_lane = np.array(list(zip(np.concatenate((left_fitx + self.window_width/2,
                                                       right_fitx[::-1] - self.window_width/2), axis=0),
                                       np.concatenate((ploty, ploty[::-1]), axis=0))), dtype='int32')

        # Curvature
        curvature = self.cal_curvature(leftx, rightx, res_yvals, ploty)
        # Offset
        offset = self.cal_offset(img, left_fitx, right_fitx)

        output = self.visualize_lanes(img, left_lane, right_lane, inner_lane, offset)

        return output, curvature, offset

    def visualize_lanes(self, img, left_lane, right_lane, inner_lane, offset):

        output = np.zeros_like(img)
        background = np.zeros_like(img)

        # Left & Right lane
        cv2.fillPoly(output, [left_lane], color=[255, 255, 0])
        cv2.fillPoly(output, [right_lane], color=[255, 255, 0])

        # Background -- if offset < 0.5m --> green, else red
        if offset < 0.55:
            background_color = [255, 255, 0]
        else:
            background_color = [255, 0, 0]
        cv2.fillPoly(background, [inner_lane], color=background_color)
        output = cv2.addWeighted(output, 1.0, background, 0.6, 0.0)

        return output

    def cal_curvature(self,  leftx, rightx, res_yvals, ploty):
        ym_per_pix = self.ym_per_pixel
        xm_per_pix = self.xm_per_pixel

        left_fit = np.polyfit(np.array(res_yvals, np.float32)*ym_per_pix, np.array(leftx, np.float32)*xm_per_pix, 2)
        right_fit = np.polyfit(np.array(res_yvals, np.float32)*ym_per_pix, np.array(rightx, np.float32)*xm_per_pix, 2)

        rad_curv_left = ((1 + (2*left_fit[0]*ploty[-300]*ym_per_pix + left_fit[1])**2)**1.5)/abs(2*left_fit[0])
        rad_curv_right = ((1 + (2 * right_fit[0] * ploty[-300] * ym_per_pix + right_fit[1]) ** 2) ** 1.5) / abs(2 * right_fit[0])

        curvature = np.mean(np.array((rad_curv_left, rad_curv_right), dtype='float32'))
        return curvature

    def cal_offset(self, img, left_fitx, right_fitx):
        camera_center = (left_fitx[-1] + right_fitx[-1])/2
        center_diff = (camera_center - img.shape[1]/2)*self.xm_per_pixel
        return center_diff


def window_mask(width, height, img, center, level):
    row = img.shape[0]
    col = img.shape[1]
    output = np.zeros_like(img)
    output[int(row - (level+1)*height):int(row - level*height), max(0, int(center-width)):min(int(center+width), col)] = 1
    return output


def draw_windows(img, w, h, window_centroids):
    # points to draw left and right windows
    left_pts = np.zeros_like(img)
    right_pts = np.zeros_like(img)

    # pixels used to find left and right lanes
    rightx = []
    leftx = []

    for level in range(0, len(window_centroids)):
        leftx.append(window_centroids[level][0])
        rightx.append(window_centroids[level][1])
        # Draw window
        l_mask = window_mask(w, h, img, window_centroids[level][0], level)
        r_mask = window_mask(w, h, img, window_centroids[level][1], level)
        # Add graphic points to window mask
        left_pts[(left_pts == 255) | (l_mask == 1)] = 255
        right_pts[(right_pts == 255) | (r_mask == 1)] = 255

    # Draw result
    template = np.array(left_pts+right_pts, dtype='uint8')
    zero_channel = np.zeros_like(template)
    template = np.array(cv2.merge((zero_channel, template, zero_channel)), dtype='uint8')
    warped = 255*np.dstack((img, img, img)).astype('uint8')
    result = cv2.addWeighted(warped, 0.3, template, 1.0, 0.)
    return result, leftx, rightx

