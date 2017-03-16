import cv2
import numpy as np
from scipy.ndimage.measurements import label


class Vehicle(object):
    # Bounding Box for Vehicle
    def __init__(self, x, y):
        self.minx = x
        self.miny = y
        self.maxx = x
        self.maxy = y
        self.points = []
        self.points.append((x, y))

    def show(self):
        return (self.minx, self.miny), (self.maxx, self.maxy)

    def add(self, x, y):
        self.points.append((x, y))
        self.minx = min(self.minx, x)
        self.miny = min(self.miny, y)
        self.maxx = max(self.maxx, x)
        self.maxy = max(self.maxy, y)


    def isNear(self, x, y):
        dist = 1000000000
        cx = max(min(x, self.maxx), self.minx)
        cy = max(min(y, self.maxy), self.miny)

        dist = self.distSqt(cx, cy, x, y)
        if dist < 50*50:
            return True
        else:
            return False

    def distSqt(self, x1, y1, x2, y2):
        delta_x = x1-x2
        delta_y = y1-y2
        dist = np.sqrt(delta_x**2 + delta_y**2)
        return dist

    def size(self):
        return (self.maxx - self.minx)*(self.maxx - self.miny)


class VehicleTracker(object):
    def __init__(self, looking_back_frames=10, distance_threshold=10):

        #  List of previous heat_maps
        self.heat_maps = []
        # How far to look back
        self.smooth_factor = looking_back_frames

        # Blob
        self.tracked_vehicles = None

    def update(self, new_heat_map, threshold=10):

        # If we are just started to recording, keep summing
        if len(self.heat_maps) < self.smooth_factor:
            if len(self.heat_maps) > 2:
                # Check False Positive.
                new_heat_map[(new_heat_map - self.heat_maps[-1]) < 0] = 0

            self.heat_maps.append(new_heat_map)
            updated_map = np.sum(self.heat_maps, axis=0) / len(self.heat_maps)

            # Remove objects that are not cars - low threshold
            updated_map[updated_map <= (threshold * (len(self.heat_maps) / self.smooth_factor))] = 0
        else:
            # Check False Positive. Look at previous frame and compare
            new_heat_map[(new_heat_map - self.heat_maps[-1]) < 0] = 0

            # Add new map to current heatmap
            self.heat_maps.append(new_heat_map)
            updated_map = np.sum(self.heat_maps, axis=0) / len(self.heat_maps)

            # Remove the earliest heat map
            earliest_map = self.heat_maps.pop(0)
            updated_map -= earliest_map

            # Remove objects that are not cars - low threshold
            updated_map[updated_map <= threshold] = 0

        vehicles = label(updated_map)

        cars = self.draw_car_box(vehicles)

        # @TODO: Update car bounding boxes

        return cars, updated_map

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

            # Determine if new box is existed in current boxes

            car_boxes.append(bbox)
        return car_boxes