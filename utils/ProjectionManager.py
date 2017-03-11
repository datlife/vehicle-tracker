import cv2
import numpy as np
from utils.CameraCalibrator import CameraCalibrator

# Define source and destination points
offset = 300


class ProjectionManager(object):
    def __init__(self, cam_calib, row, col, src=None, dst=None, offset=100):
        self.col = col
        self.row = row
        self.offset = offset
        if src is None:
            self.src = np.float32([[[col * 0.05, row],  # bottom-left
                               [col * 0.95, row],  # bottom-right
                               [col * 0.60, row * 0.62],  # top-right
                               [col * 0.43, row * 0.62]]])  # top-left
        else:
            self.src = src

        if dst is None:
            self.dst = np.float32([[col * 0.15 + offset, row],  # bottom left
                              [col * 0.90 - offset, row],  # bottom right
                              [col - offset, 0],  # top right
                              [offset, 0]])  # top le
        else:
            self.dst = dst
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inverse = cv2.getPerspectiveTransform(self.dst, self.src)
        if cam_calib is not None:
            self.mtx, self.dst, _ = cam_calib.get()

    def get_birdeye_view(self, img, size=None):

        if size is None:
            size = (self.col, self.row)
        # Warp image to a top-down view
        warped = cv2.warpPerspective(img, self.M, size, flags=cv2.INTER_LINEAR)
        return warped

    def get_normal_view(self, bird_eye_img, size=None):
        if size is None:
            size = (self.col, self.row)
        warped = cv2.warpPerspective(bird_eye_img, self.M_inverse, size, flags=cv2.INTER_LINEAR)
        return warped

    def get_roi(self):
        return self.src.astype(int)
