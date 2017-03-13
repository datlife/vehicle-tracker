import time
import json
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from yolo.net import TFNet


class SupportVectorMachineClassifier(object):

    def __init__(self):
        self.svc = Pipeline([('scaling', StandardScaler()), ('classification', LinearSVC(loss='hinge')),])

    def train(self, x_train, y_train):
        print("\nStarting to train vehicle detection classifier.")
        start = time.time()
        self.svc.fit(x_train, y_train)
        print("Completed training in {:5f} seconds.\n".format(time.time() - start))

    def score(self, x_test, y_test):
        print("Testing accuracy:")
        scores = self.svc.score(x_test, y_test)
        print("Accuracy {:3f}%".format(scores))

    def predict(self, feature):
        return self.svc.predict(feature)

    def decision_function(self, feature):
        return self.svc.decision_function(feature)


class YOLOV2(object):
    def __init__(self, cfg_path="cfg/tiny-yolo-voc.cfg", weight_path="bin/tiny-yolo-voc.weights"):
        option = {"model": cfg_path, "load": weight_path, "threshold": 0.1}
        self.model = TFNet(option)

    def train(self, img):
        raise NotImplemented

    def predict(self, img):
        result = self.model.return_predict(img)
        boxes = self._convert_json_to_points(result)
        return boxes

    def _convert_json_to_points(self, jfile):
        boxes = []
        for i in jfile:
            i = str(i).replace("\'", "\"")
            data = json.loads(i)
            top = (int(data['topleft']['x']), int(data['topleft']['y']))
            bot = (int(data['bottomright']['x']), int(data['bottomright']['y']))
            boxes.append((top, bot))
        return boxes
