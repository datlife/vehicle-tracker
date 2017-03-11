import cv2
import numpy as np


class ImageFilter(object):
    '''
    Handle image filtering
    '''

    def __init__(self, img_size):
        self.row = img_size[0]
        self.col = img_size[1]

    def abs_sobel_thresh(self, img, orient='x', thresh_min=0, thresh_max=255):
        # Define a function that applies Sobel x or y,
        # Apply the following steps to img
        # 1) Convert to grayscale
        if img.shape[-1] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        # 2, 3) Take the absolute derivative in x or y given orient = 'x' or 'y'
        abs_sobel = None
        if orient is 'x':
            abs_sobel = np.abs(cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=1, dy=0))
        if orient is 'y':
            abs_sobel = np.abs(cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=0, dy=1))
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        if abs_sobel is not None:
            scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
            # 5) Create a mask of 1's where the scaled gradient magnitude
            # masked everything to dark ( = 0)
            abs_sobel_output = np.zeros_like(scaled_sobel)
            # if any pixel has thresh_min < value < thresh_max
            abs_sobel_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
            return abs_sobel_output
        else:
            return None

    def mag_thresh(self, img, sobel_kernel=3, mag_thresh=(0, 255)):

        # Apply the following steps to img
        # 1) Convert to grayscale
        if img.shape[-1] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Calculate the magnitude
        gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

        # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scale_factor = np.max(gradient_magnitude) / 255
        gradient_magnitude = np.uint8(gradient_magnitude / scale_factor)

        # 5) Create a binary mask where mag thresholds are met
        mag_binary_output = np.zeros_like(gradient_magnitude)
        mag_binary_output[(gradient_magnitude >= mag_thresh[0]) & (gradient_magnitude <= mag_thresh[1])] = 1
        return mag_binary_output

    def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi / 2)):
        # Apply the following steps to img
        # 1) Convert to grayscale
        if img.shape[-1] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Take the absolute value of the x and y gradients
        abs_sobelx = np.abs(sobelx)
        abs_sobely = np.abs(sobely)
        # 4) Calculate the direction of the gradient
        grad_direction = np.arctan2(abs_sobely, abs_sobelx)

        # 5) Create a binary mask where direction thresholds are met
        dir_binary_output = np.zeros_like(grad_direction)
        # 6) Return this mask as your binary_output image
        dir_binary_output[(grad_direction >= thresh[0]) & (grad_direction <= thresh[1])] = 1
        return dir_binary_output

    def hls_select(self, img, thresh=(0, 255), channel=2):
        # 1) Convert to HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        # 2) Apply a threshold to the S channel
        s_channel = hls[:, :, channel]  # default is s_channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel > thresh[0]) & (s_channel < thresh[1])] = 1
        # 3) Return a binary image of threshold result
        binary_output = s_binary  # placeholder line
        return binary_output

    def adaptive_equalize_image(self, img, level):
        """
        Equalize an image - Increase contrast for the image
            # http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
        :param img:    an gray image
        :param level:  clipLevel
        :return: a equalized image
        """
        # Conver BGR to LAB
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=level)
        cl = clahe.apply(l)
        merge = cv2.merge((cl, a, b))
        result = cv2.cvtColor(merge, cv2.COLOR_LAB2BGR)
        return result

    def mix_threshold(self, img):
        #
        # @TODO:
        # Find a way to analyze image and adjust image robustly:
        # Cloudy --> what
        # Shadow Tree ---> what
        # Overexposure --> what
        #
        # Sobel Threshold
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gradx = self.abs_sobel_thresh(gray, orient='x', thresh_min=30, thresh_max=100)
        grady = self.abs_sobel_thresh(gray, orient='y', thresh_min=30, thresh_max=100)
        mag_binary = self.mag_thresh(gray, mag_thresh=(30, 100), sobel_kernel=15)
        dir_binary = self.dir_threshold(gray, thresh=(0.7, 1.3), sobel_kernel=15)

        # Color Threshold
        s_binary = self.hls_select(img, thresh=(88, 250))
        h_binary = self.hls_select(img, thresh=(120, 250), channel=1)
        color_binary = np.zeros_like(s_binary)
        color_binary[(s_binary == 1) & (h_binary == 1)] = 1

        # Mix Threshold together
        combined_binary = np.zeros_like(gray)
        combined_binary[((gradx == 1) | (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (color_binary == 1)] = 1

        return combined_binary

    def region_of_interest(self, img, vertices):
        """
        Filter out not-so-important region in the image
        :param source_img:
        :param vertices:    list of vertices to create a polygon
        :return:
        """
        mask = np.zeros_like(img)
        ignore_mask_color = 255

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        cv2.fillPoly(mask, vertices, ignore_mask_color)

        masked_edges = cv2.bitwise_and(img, mask)
        return masked_edges

    def draw_line_segments(self, source_image, h_lines, color=[255, 0, 0], thickness=2):
        """
        Draw the line segments to the source images.
        """

        line_img = np.copy(source_image)
        for a_line in h_lines:
            for x1, y1, x2, y2 in a_line:
                cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
        return line_img

    def mix_color_grad_thresh(self, img, grad_thresh=(30, 100), s_thresh=(88, 250), h_thresh=(120, 250), dir_thresh=(0.7, 1.4)):
        # THIS IS FASTER
        # Convert to HLS color space and separate the S channel
        # Note: img is the undistorted image
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]
        h_channel = hls[:, :, 1]
        # Grayscale image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Sobel x
        abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))  # Take the derivative in x
        abs_sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))  # Take the derivative in x
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        scaled_sobel_y = np.uint8(255 * abs_sobely / np.max(abs_sobely))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= grad_thresh[0]) & (scaled_sobel <= grad_thresh[1])] = 1
        sxbinary[(scaled_sobel_y >= grad_thresh[0]) & (scaled_sobel_y <= grad_thresh[1])] = 1

        # Threshold magnitude gradient
        gradient_magnitude = np.sqrt(abs_sobelx ** 2 + abs_sobely ** 2)
        scale_factor = np.max(gradient_magnitude) / 255
        gradient_magnitude = np.uint8(gradient_magnitude / scale_factor)
        mag_binary = np.zeros_like(gradient_magnitude)
        mag_binary[(gradient_magnitude >= grad_thresh[0]) & (gradient_magnitude <= grad_thresh[1])] = 1

        # Threshold direction gradient
        grad_direction = np.arctan2(abs_sobely, abs_sobelx)
        dir_binary = np.zeros_like(grad_direction)
        dir_binary[(grad_direction >= dir_thresh[0]) & (grad_direction <= dir_thresh[1])] = 1

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1]) &
                 (h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1

        # Combine the two binary thresholds
        # OR mean combine
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(sxbinary == 1) | (s_binary == 1) | ((mag_binary == 1) & (dir_binary == 1))] = 1

        return combined_binary