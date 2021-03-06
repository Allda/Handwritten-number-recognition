########################################################
# Authors: Ales Raszka, Marek Fiala, Matus Dobrotka
# Project: POV
# Year: 2016
########################################################


import numpy as np
import time
import sys

import cv2

from constants import LINEAR
from constants import K_NEAREST
from constants import ADABOOST
from constants import RANDOM_FOREST
from constants import POLYNOMIAL


class openCVClassifier(object):
    """This object provide classification methods from OpenCV"""

    classifier_file_name = 'classifier-opencv.pkl'

    def __init__(self):
        self.hog = None
        self.classifier = None
        self.type = ""

    def init_hog(self, win_size=(28, 28), block_size=(14, 14), cell_size=(14, 14), nbins=9):
        """Initialize parameters for HOGDetector.
        Block stride is same as cell size because of compatibility with another library """
        block_stride = cell_size
        self.hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

    def create_classifier(self, name):
        """Create classifier which is specified by name"""
        if name == LINEAR:
            self.classifier = cv2.ml.SVM_create()
            self.classifier.setType(cv2.ml.SVM_C_SVC)
            self.classifier.setGamma(0.1)
            self.classifier.setC(10)
            self.classifier.setKernel(cv2.ml.SVM_LINEAR)
        elif name == POLYNOMIAL:
            self.classifier = cv2.ml.SVM_create()
            self.classifier.setType(cv2.ml.SVM_C_SVC)
            self.classifier.setC(0.1)
            self.classifier.setDegree(3)
            self.classifier.setKernel(cv2.ml.SVM_POLY)
        elif name == K_NEAREST:
            self.classifier = cv2.ml.KNearest_create()
        elif name == ADABOOST:
            print "AdaBoost is not implemented in OpenCV for more than two classes."
            sys.exit(1)
            pass
        elif name == RANDOM_FOREST:
            self.classifier = cv2.ml.RTrees_create()
            self.classifier.setMaxDepth(11)
            self.classifier.setMinSampleCount(5)
            self.classifier.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01))
        else:
            print "No correct classifier set: %s" % name

        self.type = name

    def load_classifier(self):
        """In OpenCV 3.1 there is no possibility to load classifier from image yet"""
        pass

    def train_classifier(self, training_imgs, training_labels):
        """Train classifier on added data"""
        image_hog_list = []
        start = time.time()
        for image in training_imgs:
            image_hog = self.get_hog_for_img(image.reshape((28, 28)))
            image_hog_list.append(image_hog)

        end = time.time()
        print("Time to calculation of HOG per image: " + str((end - start)/len(image_hog_list)))
        print "HOG feature vector length: " + str(len(image_hog_list[0]))

        hog_features = np.array(image_hog_list, 'float32')

        start = time.time()
        if self.type == LINEAR or self.type == RANDOM_FOREST:
            self.classifier.train(hog_features, cv2.ml.ROW_SAMPLE, np.array(training_labels, dtype=np.int32))
        else:
            self.classifier.train(hog_features, cv2.ml.ROW_SAMPLE, training_labels)

        end = time.time()
        print("time to train: " + str(end - start))

    def test_classifier(self, testing_imgs, testing_labels):
        """Test classifier on added data"""
        image_hog_list_test = []
        start = time.time()
        for image in testing_imgs:
            image_hog = self.get_hog_for_img(image.reshape((28, 28)))
            image_hog_list_test.append(image_hog)

        hog_features_test = np.array(image_hog_list_test, 'float32')

        result = []
        if self.type == K_NEAREST:
            ret, result, neighbours, dist = self.classifier.findNearest(hog_features_test, k=5)
        else:
            result = self.classifier.predict(hog_features_test)
            result = result[1]

        end = time.time()
        print("time to test: " + str(end - start))

        correct = 0
        i = 0
        for row in result:
            if row == testing_labels[i]:
                correct += 1

            i += 1

        accuracy = correct * 100.0 / result.size
        print accuracy

    def classify_img(self, img):
        """Classify number which is written in image"""
        image_hog = self.get_hog_for_img(img.reshape((28, 28)))

        if self.type == K_NEAREST:
            ret, result, neighbours, dist = self.classifier.findNearest([image_hog], k=5)
        else:
            result = self.classifier.predict([image_hog])
            result = result[1]

        return result

    def get_hog_for_img(self, img):
        return self.hog.compute(np.uint8(img))

    def save_classifier(self):
        self.classifier.save(self.classifier_file_name)
        print "Classifer was save to file: %s" % self.classifier_file_name
