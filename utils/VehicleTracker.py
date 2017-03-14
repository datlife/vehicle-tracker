import cv2
import numpy as np
from scipy.ndimage.measurements import label


class Vehicle(object):
    # Bounding Box for Vehicle
    xmin = []
    xmax = []
    ymin  = []
    ymax = []
    def __init__(self):
        # How many previous frames will be smooth
        self.smooth_factor = 15


class VehicleTracker(object):
    def __init__(self, looking_back_frames=10):
        # A list of bounding boxes for each vehicles
        self.tracked_vehicles = []
        # Current heat_map
        self.heat_map = []
        # How far to look back
        self.smooth_factor = looking_back_frames

    def update(self, new_heat_map, threshold=10):

        # If we are just started to recording, keep averaging
        if len(self.heat_map) < self.smooth_factor:
            self.heat_map.append(new_heat_map)
            updated_map = np.sum(self.heat_map, axis=0)
        else:
            # Remove the earliest frame
            self.heat_map.pop(-1)
            # Add new map
            self.heat_map.append(new_heat_map)
            # Average out the heat map
            updated_map = np.sum(self.heat_map, axis=0)

        # Remove previous frame
        updated_map[updated_map >= (np.max(updated_map)-1)] = 0
        # Remove objects that are not car
        updated_map = self.filter_out_not_cars(updated_map, threshold=threshold)
        # Create an heat image
        img = 255*updated_map/np.max(updated_map)
        img = np.dstack((img, updated_map, updated_map)).astype(np.uint8)

        labels = label(updated_map)
        cars = self.draw_car_box(labels)

        return cars, img


    def filter_out_not_cars(self, heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return threshold-ed map
        return heatmap

    def draw_car_box(self, labels):
        # Iterate through all detected cars
        car_boxes = []
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            car_boxes.append(bbox)
        return car_boxes
