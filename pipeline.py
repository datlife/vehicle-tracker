import cv2
import numpy as np
import itertools
from utils import SupportVectorMachineClassifier, YOLOV2
from utils import VehicleTracker
from utils import get_file_names, find_cars, get_feature
# Import video and process
from moviepy.editor import VideoFileClip

ystart = 400
ystop = 656
scale = 1.75


def process_image(frame):
    global clf
    global car_tracker
    global ystar, ystop, scale

    svc_img = np.copy(frame)
    heatmap, windows = find_cars(frame, ystart, ystop, scale, clf, dec_thresh=0.99)
    cars, heatmap = car_tracker.update(heatmap, threshold=2.0)

    # for p1, p2 in itertools.chain(cars):
    #     # Draw SVC boxes
    #     cv2.rectangle(svc_img, p1, p2, (255, 255, 0), 3)

    svc_img = cv2.addWeighted(svc_img, 1.0, heatmap, 0.8, 0.0)
    return svc_img

if __name__ == "__main__":
    # Import car and not car images
    cars = get_file_names('./data/vehicles', pattern='*.png')
    not_cars = get_file_names('./data/non-vehicles', pattern='*.png')

    # Calculate car features & not-car features
    car_features = get_feature(cars, workers=4)
    not_car_features = get_feature(not_cars, workers=4)

    # Create data set
    x = np.vstack((car_features, not_car_features)).astype(np.float64)
    y = np.concatenate((np.ones(len(car_features)), np.zeros(len(not_car_features))))

    # SVC classifier
    clf = SupportVectorMachineClassifier()
    clf.train(x, y)

    # Vehicle Tracker
    car_tracker = VehicleTracker(looking_back_frames=10)

    output = 'output.mp4'
    clip1 = VideoFileClip("./project_video.mp4").subclip(14, 22)
    clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    clip.write_videofile(output, audio=False)


