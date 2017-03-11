import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    draw_img = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return draw_img


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

#
# image = mpimg.imread('bbox-example-image.jpg')
# # Here are the bounding boxes I used
# bboxes = [((275, 572), (380, 510)), ((488, 563), (549, 518)), ((554, 543), (582, 522)),
#           ((601, 555), (646, 522)), ((657, 545), (685, 517)), ((849, 678), (1135, 512))]
#
# result = draw_boxes(image, bboxes)
# plt.imshow(result)
# plt.show()
