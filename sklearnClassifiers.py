import numpy as np
import time
import json

from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

from constants import LINEAR
from constants import K_NEAREST
from constants import ADABOOST
from constants import K_MEANS
from constants import RANDOM_FOREST


class sklearnClassifier(object):

    classifier_file_name = 'classifier-sklearn.pkl'
    pixels_per_cell = (6, 6)
    cells_per_block = (4, 4)

    def __init__(self):
        self.hog = None
        self.classifier = None
        self.type = ""

    def create_classifier(self, name):
        if name == LINEAR:
            self.classifier = LinearSVC()
        elif name == K_NEAREST:
            self.classifier = KNeighborsClassifier()
        elif name == ADABOOST:
            self.classifier = AdaBoostClassifier(DecisionTreeClassifier(),
                           algorithm="SAMME",
                           n_estimators=200)
        elif name == RANDOM_FOREST:
            self.classifier = RandomForestClassifier(n_estimators = 100)
        elif name == K_MEANS:
            self.classifier = KMeans(n_clusters=10) # We have ten numbers
        else:
            print "No correct classifier set: %s" % name

        self.type = name

    def load_classifier(self):
        try:
            self.classifier = joblib.load(self.classifier_file_name)
        except IOError:
            print "No classifier file has been found: %s" % self.classifier_file_name

    def train_classifier(self, images, labels):

        image_hog_list = []
        start = time.time()
        for image in images:
            image_hog = hog(image.reshape((28, 28)), orientations=9,
                            pixels_per_cell=self.pixels_per_cell, cells_per_block=self.cells_per_block,
                            visualise=True)
            image_hog_list.append(image_hog)
        hog_features = np.array(image_hog_list, 'float64')

        self.classifier.fit(hog_features, labels)

        end = time.time()
        print("time to train: " + str(end - start))

    def test_classifier(self, images, labels):
        result = []
        start = time.time()
        for index, image in enumerate(images):
            image_hog = hog(image.reshape((28, 28)), orientations=9,
                            pixels_per_cell=self.pixels_per_cell, cells_per_block=self.cells_per_block,
                            visualise=False)

            predicted_value = self.classifier.predict(np.array([image_hog]))[0]
            result.append((labels[index], predicted_value))

        end = time.time()
        print("time to test: " + str(end - start))

        correct = 0
        for item in result:
            if item[0] == item[1]:
                correct += 1

        print "%s %s %s" % (correct, len(result),
                            float(correct)/len(result) * 100)
        self.print_statistics(result)

    def print_statistics(self, result):
        numbers = []
        for index in range(0, 10):
            numbers.append({})
            numbers[index] = {
                'number': index,
                'total': 0,
                'correct': 0,
                'incorrect': {},
                'success_rate': 0
            }
            for incorrect_index in range(0, 10):
                numbers[index]['incorrect'][incorrect_index] = 0
        for item in result:
            numbers[item[0]]['total'] += 1
            if item[0] == item[1]:
                numbers[item[0]]['correct'] += 1
            else:
                numbers[item[0]]['incorrect'][item[1]] += 1

        for number in numbers:
            number['success_rate'] = float(number['correct']) / number['total'] * 100
        print json.dumps(numbers, indent=4)

    def get_hog_for_img(self, img):
        return self.hog.compute(np.uint8(img))

    def save_classifier(self):
        joblib.dump(self.classifier, self.classifier_file_name, compress=3)
        print "Classifer was save to file: %s" % self.classifier_file_name
