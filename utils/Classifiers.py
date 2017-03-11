import numpy as np
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class SVC(object):

    def __init__(self, x, y, test_split=0.01):
        self.svc = LinearSVC()

        # Apply Standard Scalars to normalize vector
        self.std_scaler = StandardScaler().fit(x)
        scaled_x = self.std_scaler.transform(x)

        # Split data: Training Set, Test Set
        random_state = np.random.randint(0, 100)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(scaled_x, y, test_size=0.1, random_state=random_state)
        print("Training data: Features {}, Labels {}".format(len(self.x_train), len(self.y_train)))
        print("Test data: Features {}, Labels {}".format(len(self.x_test), len(self.y_test)))

    def train(self):
        print("\nStarting to train vehicle detection classifier.")
        start = time.time()
        self.svc.fit(self.x_train, self.y_train)
        print("Completed training in {:5f} seconds.\n".format(time.time() - start))

    def score(self):
        print("Testing accuracy:")
        scores = self.svc.score(self.x_test, self.y_test)
        print("Accuracy {:3f}%".format(scores))

    def predict(self, feature):
        '''
        Return 1 or 0
        :param feature:
        :return:
        '''
        return self.svc.predict(feature)
