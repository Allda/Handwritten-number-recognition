import numpy as np
import time
import json

from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

from constants import LINEAR
from constants import K_NEAREST
from constants import ADABOOST
from constants import RANDOM_FOREST
from constants import POLYNOMIAL


class sklearnClassifier(object):

    classifier_file_name = 'classifier-sklearn.pkl'
    pixels_per_cell = (14, 14)
    cells_per_block = (2, 2)
    win_size = (28, 28)
    nbins = 9

    def __init__(self):
        self.classifier = None
        self.type = ""

    def init_hog(self, win_size=(28, 28), block_size=(14, 14), cell_size=(14, 14), nbins=9):
        self.win_size = win_size
        self.pixels_per_cell = cell_size
        self.cells_per_block = (block_size[0]/cell_size[0], block_size[1]/cell_size[1])
        print self.pixels_per_cell
        print self.cells_per_block
        self.nbins = nbins

    def create_classifier(self, name):
        if name == LINEAR:
            self.classifier = SVC(kernel="linear", gamma=0.1, C=10)
        elif name == POLYNOMIAL:
            self.classifier = SVC(kernel="poly", C=0.1, degree=3)
        elif name == K_NEAREST:
            self.classifier = KNeighborsClassifier()
        elif name == ADABOOST:
            self.classifier = AdaBoostClassifier(DecisionTreeClassifier(),
                           algorithm="SAMME",
                           n_estimators=200)
        elif name == RANDOM_FOREST:
            self.classifier = RandomForestClassifier(n_estimators=100, max_depth=11, min_samples_split=5)
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
            image_hog = self.get_hog_for_img(image.reshape((28, 28)))

            image_hog_list.append(image_hog)

        end = time.time()
        print("Time to calculation of HOG per image: " + str((end - start)/len(image_hog_list)))
        print "HOG feature vector length: " + str(len(image_hog_list[0]))

        hog_features = np.array(image_hog_list, 'float64')

        start = time.time()
        self.classifier.fit(hog_features, labels)

        end = time.time()
        print("time to train: " + str(end - start))

    def test_classifier(self, images, labels):
        result = []
        start = time.time()
        for index, image in enumerate(images):
            image_hog = self.get_hog_for_img(image.reshape((28, 28)))

            predicted_value = self.classifier.predict(np.array([image_hog]))[0]
            result.append((labels[index], predicted_value))

        end = time.time()
        print("time to test: " + str(end - start))

        correct = 0
        for item in result:
            if item[0] == item[1]:
                correct += 1

        # print "%s %s %s" % (correct, len(result),
        #                     float(correct)/len(result) * 100)
        print float(correct)/len(result) * 100
        # self.print_statistics(result)

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

    def classify_img(self, img):
        roi_hog_fd = self.get_hog_for_img(img)
        result = self.classifier.predict(np.array([roi_hog_fd], 'float64'))

        print result

        return result

    def get_hog_for_img(self, img):
        return hog(img, orientations=self.nbins,
                        pixels_per_cell=self.pixels_per_cell, cells_per_block=self.cells_per_block,
                        visualise=False)

    def save_classifier(self):
        joblib.dump(self.classifier, self.classifier_file_name, compress=3)
        print "Classifer was save to file: %s" % self.classifier_file_name
