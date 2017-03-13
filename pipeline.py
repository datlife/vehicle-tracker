import cv2
import numpy as np
import itertools
from utils import SupportVectorMachineClassifier, YOLOV2
from utils import get_feature, get_file_names, search_windows, slide_window, draw_boxes
# Import video and process
from moviepy.editor import VideoFileClip


def process_image(frame):
    global clf
    global yolo
    global windows

    svc_frame = frame.astype(np.float32)
    # VEHICLE TRACKER
    # heatmap = np.zeros_like(svc_img[:, :, 0])
    # positive_windows = search_windows(svc_frame, windows, clf, size=(64, 64), decision_threshold=dec_thresh)
    positive_windows = yolo.predict(frame)
    result = draw_boxes(frame, positive_windows)
    return result

if __name__ == "__main__":
    # Import car and not car images
    cars = get_file_names('./data/vehicles', pattern='*.png')
    not_cars = get_file_names('./data/non-vehicles', pattern='*.png')

    # Calculate car features & not-car features
    # car_features = get_feature(cars, workers=4)
    # not_car_features = get_feature(not_cars, workers=4)

    # Create data set
    # x = np.vstack((car_features, not_car_features)).astype(np.float64)
    # y = np.concatenate((np.ones(len(car_features)), np.zeros(len(not_car_features))))

    dec_thresh = 0.75
    window_size = (96, 96)
    x_region = [None, None]
    y_region = [400, None]
    over_lap = (0.7, 0.7)
    img_size = (720, 1280)

    # Create SVC Classifier
    clf = SupportVectorMachineClassifier()
    yolo = YOLOV2()

    # Train classifier
    # clf.train(x, y)

    # Create sliding windows
    windows = slide_window(img_size,
                           x_start_stop=x_region,
                           y_start_stop=y_region,
                           xy_window=window_size,
                           xy_overlap=over_lap)

    output = 'output.mp4'
    clip1 = VideoFileClip("./project_video.mp4").subclip(6, 12)
    clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    clip.write_videofile(output, audio=False)


