import numpy as np

import cv2

LINEAR = "linear"
K_MEANS = "k-means"
K_NEAREST = "k-nearest"
ADABOOST = "adaboost"

class openCVClassifier(object):

    classifier_file_name = 'classifier-opencv.pkl'

    def __init__(self):
        self.hog = None
        self.classifier = None
        self.type = ""

    def create_hog(self, win_size = (28, 28), block_size = (14, 14), block_stride = (14, 14), cell_size = (14, 14), nbins = 9):
        self.hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

    def create_classifier(self, name):
        if name == LINEAR:
            self.classifier = cv2.ml.SVM_create()
            self.classifier.setType(cv2.ml.SVM_C_SVC)
            self.classifier.setGamma(5.383)
            self.classifier.setC(2.67)
            self.classifier.setKernel(cv2.ml.SVM_LINEAR)
        elif name == K_NEAREST:
            self.classifier = cv2.ml.KNearest_create()
        elif name == ADABOOST:
            pass
        elif name == K_MEANS:
            pass
        else:
            print "No correct classifier set: %s" % name

        self.type = name

    def train_classifier(self, training_imgs, training_labels):
        image_hog_list = []
        for image in training_imgs:
            image_hog = self.get_hog_for_img(image.reshape((28, 28)))
            image_hog_list.append(image_hog)

        hog_features = np.array(image_hog_list, 'float32')

        if self.type == LINEAR:
            self.classifier.train(hog_features, cv2.ml.ROW_SAMPLE, np.array(training_labels, dtype=np.int32))
        else:
            self.classifier.train(hog_features, cv2.ml.ROW_SAMPLE, training_labels)

    def test_classifier(self, testing_imgs, testing_labels):
        image_hog_list_test = []
        for image in testing_imgs:
            image_hog = self.get_hog_for_img(image.reshape((28, 28)))
            image_hog_list_test.append(image_hog)

        hog_features_test = np.array(image_hog_list_test, 'float32')

        result = []
        if self.type == K_NEAREST:
            ret, result, neighbours, dist = self.classifier.findNearest(hog_features_test, k=5)
        elif self.type == LINEAR:
            result = self.classifier.predict(hog_features_test)
            result = result[1]

        correct = 0
        i = 0
        for row in result:
            if row == testing_labels[i]:
                correct += 1

            i += 1

        accuracy = correct * 100.0 / result.size
        print accuracy

    def get_hog_for_img(self, img):
        return self.hog.compute(np.uint8(img))

    def save_classifier(self):
        self.classifier.save(self.classifier_file_name)
        print "Classifer was save to file: %s" % self.classifier_file_name
