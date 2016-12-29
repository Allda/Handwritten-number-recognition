import numpy as np

import json

from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.externals import joblib

LINEAR = "linear"
K_MEANS = "k-means"
K_NEAREST = "k-nearest"
ADABOOST = "adaboost"


class sklearnClassifier(object):

    classifier_file_name = 'classifier-sklearn.pkl'

    def __init__(self):
        self.hog = None
        self.classifier = None
        self.type = ""

    # def create_hog(self, win_size = (28, 28), block_size = (14, 14), block_stride = (14, 14), cell_size = (14, 14), nbins = 9):

    def create_classifier(self, name):
        if name == LINEAR:
            self.classifier = LinearSVC()
        elif name == K_NEAREST:
            self.classifier = KNeighborsClassifier()
        elif name == ADABOOST:
            self.classifier = AdaBoostClassifier(DecisionTreeClassifier(),
                           algorithm="SAMME",
                           n_estimators=200)
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
        for image in images:
            image_hog = hog(image.reshape((28, 28)), orientations=9,
                            pixels_per_cell=(14, 14), cells_per_block=(1, 1),
                            visualise=False)
            image_hog_list.append(image_hog)
        hog_features = np.array(image_hog_list, 'float64')

        self.classifier.fit(hog_features, labels)

    def test_classifier(self, images, labels):
        result = []
        for index, image in enumerate(images):
            image_hog = hog(image.reshape((28, 28)), orientations=9,
                            pixels_per_cell=(14, 14), cells_per_block=(1, 1),
                            visualise=False)
            predicted_value = self.classifier.predict(np.array([image_hog]))[0]
            result.append((labels[index], predicted_value))

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
