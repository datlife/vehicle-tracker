import cv2
import numpy as np
import os
import glob
import pickle


class CameraCalibrator(object):
    '''
    Calibrate distortion of a camera given a set of chess images
    '''
    def __init__(self, img_dir=None, p_file=None, nx=9, ny=6):

        self.camera_matrix = None
        self.distortion_coefficients = None
        self.nx = nx        # Number of corners in a row
        self.ny = ny        # Number of corners in a column
        self.img_size = None
        self.obj_points = []  # 3D representation of a point
        self.img_points = []  # 2D representation of a point
        self.calibrate_done = False
        if p_file is not None:
            self.import_pickle(p_file)
        if (img_dir is not None) and p_file is None:
            self.run(img_dir)

    def run(self, img_dir):

        print("Calibrating camera...\n")
        # Create a list of images file path
        images = glob.glob(img_dir + "/calibration*.jpg")
        chessboard_size = (self.nx, self.ny)

        # Create a list of object_points
        obj_pts = np.zeros((self.nx*self.ny, 3), dtype='float32')
        obj_pts[:, :2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1, 2)

        # Iterate through each image to find corners
        for i, img_path in enumerate(images):
            img = cv2.imread(img_path)
            self.img_size = (img.shape[0], img.shape[1])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found_corners, corners = cv2.findChessboardCorners(gray, chessboard_size)
            if found_corners:
                self.obj_points.append(obj_pts)
                self.img_points.append(corners)
        # Calibrate camera
        okay, self.camera_matrix, self.distortion_coefficients, _, _ = cv2.calibrateCamera(self.obj_points, self.img_points,
                                                                                           self.img_size, None, None)

        if okay:
            print("Calibrate camera successfully.\n")
            print("Camera matrix : \n{}\nDistortion_coeffs: \n{}\n".format(self.camera_matrix, self.distortion_coefficients))
            self.calibrate_done = True
        else:
            print("Unknown issue during calibration.\n")

    def import_pickle(self, pickle_file):
        if self.calibrate_done is False:
            with open(pickle_file, 'rb') as f:
                pickle_data = pickle.load(f)
                self.img_size = pickle_data['img_size']
                self.camera_matrix = pickle_data['mtx']
                self.distortion_coefficients = pickle_data['dist']
                del pickle_data
                print("Camera Calibration data restored from", pickle_file)
        else:
            print("Calibration is finished. Cannot import!")

    def export_pickle(self, pickle_file):
        if self.calibrate_done:
            try:
                with open(pickle_file, 'w+b') as pfile:
                    print("Saving calibration matrix into file {}".format(pickle_file))
                    data = {'mtx': self.camera_matrix,
                            'dist': self.distortion_coefficients,
                            'img_size': self.img_size}
                    pickle.dump(data, pfile, protocol=2)
            except Exception as e:
                print("Unable to save data due to error {}".format(e))
            finally:
                pass

    def get(self):
        return self.camera_matrix, self.distortion_coefficients, self.img_size
