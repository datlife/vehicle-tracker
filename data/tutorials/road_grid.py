import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from utils import draw_windows
from utils import ProjectionManager, LineTracker, ImageFilter


def draw_line_segments(source_image, lines, color=[255, 0, 0], thickness=2):
    """
    Draw the line segments to the source images.
    """
    line_img = np.copy(source_image)
    for  left, right in lines:
            cv2.line(line_img, (left[0],left[1]), (right[0],right[1]), color, thickness)
    return line_img


def create_windows(row, col, left_line, right_line, y_step=20):
    '''
    Create a list of windows in bird eye images using left and right line as the boundary
    :param left_line:
    :param right_line:
    :return:
    '''
    # Calculate the width of each windows
    len_lines = np.mean(right_line) - np.mean(left_line)

    # Number of window lines in each lane
    num_lines = np.int(col / y_step)
    lines = []
    for ys in range(num_lines):
        left_y = ys * y_step
        left_x = left_line[left_y, :][0]
        right_y = ys * y_step
        right_x = right_line[right_y, :][0]
        lines.append(((left_x, left_y), (right_x, right_y)))

    return lines


# Try different approach in Project 4
curve_centers = LineTracker(window_height=80, window_width=20, margin=25, ym=10 / 720, xm=4 / 384, smooth_factor=1)


def lane_detection(frame):
    global curve_centers
    r, w, c = frame.shape
    img_filter = ImageFilter((r, w))
    # NOTICE:
    # Offset will scale the output of bird-eye image (zoomin/out).
    # If all lanes are not detected, adjust offset at larger value to get all lanes
    projmgr = ProjectionManager(cam_calib=None, row=r, col=w, src=None, dst=None, offset=400)
    # I did not use cam calibration on this project
    undst_img = frame
    # Make a lane region brighter
    # undst_roi = img_filter.region_of_interest(undst_img, projmgr.get_roi())
    undst_roi = undst_img
    undst_img = cv2.addWeighted(undst_img, 0.5, undst_roi, 0.7, 0.)

    # Create a binary image
    bin_img = img_filter.mix_color_grad_thresh(undst_img, grad_thresh=(30, 100), s_thresh=(88, 250),
                                               h_thresh=(120, 250))

    # Perspective Transform
    binary_roi = img_filter.region_of_interest(bin_img, projmgr.get_roi())
    birdseye_view = projmgr.get_birdeye_view(undst_img)
    birdseye_img = projmgr.get_birdeye_view(binary_roi)
    histogram = np.sum(birdseye_img[birdseye_img.shape[0] / 2:, :], axis=0)

    # # Sliding window
    window_centroids = curve_centers.find_lane_line(warped=birdseye_img)
    windows, left_x, right_x = draw_windows(birdseye_img, w=25, h=80, window_centroids=window_centroids)

    # Curve-fit and calculate curvature and offset
    curved_fit, lines, lanes = curve_centers.curve_fit(windows, (left_x, right_x), transparency=0.4)

    # Create a list of windows
    for lane in zip(lanes, lanes[1:]):
        l = create_windows(1280, 720, lane[0], lane[1], y_step=40)
        curved_fit = draw_line_segments(curved_fit, l, color=[255, 255, 0], thickness=2)

    # Convert back to normal vieward
    lane_lines = projmgr.get_normal_view(curved_fit)

    # Merge to original image
    lane_lines = cv2.addWeighted(undst_img, 1.0, lane_lines, 0.5, 0.0)
    return undst_img, lane_lines, windows, curved_fit, birdseye_view, histogram, lines, lanes


images = glob.glob('../test_images/*.jpg')[:5]
size = len(images)
print(size)
f, ax = plt.subplots(size, 4, figsize=(30, 50))
f.tight_layout()
ax[0, 0].set_title('Original Image')
ax[0, 1].set_title('Bird Eye')
ax[0, 2].set_title('Windows')
lines = []
lanes = []
results = []
for idx, img in enumerate(images):
    frame = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
    undst_img, lane_lines, windows, curved_fit, birdseye_view, histogram, lines, lanes = lane_detection(frame)
    results.append(lane_lines)
    ax[idx, 0].imshow(undst_img)
    ax[idx, 1].imshow(birdseye_view)
    ax[idx, 2].imshow(curved_fit)
    ax[idx, 3].imshow(lane_lines)

plt.show()