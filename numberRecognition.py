import sys
import json
import argparse
import numpy as np

from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn import neighbors

from sklearn.externals import joblib

from mnist import MNIST
from ownNumberProcessor import OwnNumberProcessor

classifier_file_name = 'classifier.pkl'

def setup_parser():
    parser = argparse.ArgumentParser(description='Hand writen number '
                                                 ' recognition')
    main_group = parser.add_mutually_exclusive_group()
    main_group.add_argument('--train-linear-classifier', action='store_true',
                            help='Train linear classifier')
    main_group.add_argument('--train-k-nearest-classifier', action='store_true',
                            help='Train k-nearest classifier')
    main_group.add_argument('--classify-mnist', action='store_true',
                            help='Classify MNIST database')
    main_group.add_argument('--classify-own', default=None, metavar='FILE',
                            help='Clasify own picture with handwritten '
                                 'numbers')
    parser.add_argument('--mnist', default=None, nargs='?',
                        help='Location of MNIST database. In case it is '
                             'missing, it will try to download database')
    parser.add_argument('--classifier-file', default=None,
                        help='Location of classifier file')
    return parser


def train_linear_classifier(images, labels):
    image_hog_list = []
    for image in images:
        image_hog = hog(image.reshape((28, 28)), orientations=9,
                        pixels_per_cell=(14, 14), cells_per_block=(1, 1),
                        visualise=False)
        image_hog_list.append(image_hog)
    hog_features = np.array(image_hog_list, 'float64')

    lcvs = LinearSVC()
    lcvs.fit(hog_features, labels)

    return lcvs

def train_kNearest_classifier(images, labels):
    image_hog_list = []
    for image in images:
        image_hog = hog(image.reshape((28, 28)), orientations=9,
                        pixels_per_cell=(14, 14), cells_per_block=(1, 1),
                        visualise=False)
        image_hog_list.append(image_hog)
    hog_features = np.array(image_hog_list, 'float64')

    knn = neighbors.KNeighborsClassifier()
    knn.fit(hog_features, labels)

    return knn


def classify_MNIST_data(images, labels, classifier):
    result = []
    for index, image in enumerate(images):
        image_hog = hog(image.reshape((28, 28)), orientations=9,
                        pixels_per_cell=(14, 14), cells_per_block=(1, 1),
                        visualise=False)
        predicted_value = classifier.predict(np.array([image_hog]))[0]
        result.append((labels[index], predicted_value))
    return result

def print_statistics(result):
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


def main():
    parser = setup_parser()
    args = parser.parse_args()
    if args.classifier_file:
        global classifier_file_name
        classifier_file_name = args.classifier_file

    mnist = MNIST()

    if args.train_linear_classifier:
        print 'train linear'
        training_imgs, training_labels = mnist.load_training()
        training_imgs = np.array(training_imgs)
        lcvs = train_linear_classifier(training_imgs, training_labels)
        joblib.dump(lcvs, classifier_file_name, compress=3)
        print "Classifer was save to file: %s" % classifier_file_name
    elif args.train_k_nearest_classifier:
        print 'train k-nearest'
        training_imgs, training_labels = mnist.load_training()
        training_imgs = np.array(training_imgs)
        lcvs = train_kNearest_classifier(training_imgs, training_labels)
        joblib.dump(lcvs, classifier_file_name, compress=3)
        print "Classifer was save to file: %s" % classifier_file_name
    elif args.classify_mnist:
        print 'clasify mnist'
        try:
            classifier = joblib.load(classifier_file_name)
        except IOError:
            print "No classifier file has been found: %s" % classifier_file_name
            sys.exit(-1)
        testing_imgs, testing_labels = mnist.load_testing()
        testing_imgs = np.array(testing_imgs)
        result = classify_MNIST_data(testing_imgs, testing_labels, classifier)
        correct = 0
        for item in result:
            if item[0] == item[1]:
                correct += 1
        print "%s %s %s" % (correct, len(result),
                            float(correct)/len(result) * 100)
        print_statistics(result)
    elif args.classify_own:
        print 'classify own picture'
        print args.classify_own
        try:
            lcvs = joblib.load(classifier_file_name)
        except IOError:
            print "No classifier file has been found: %s" % classifier_file_name
            sys.exit(-1)
        processor = OwnNumberProcessor()
        processor.set_img(args.classify_own)
        processor.process(lcvs)
        # TODO: call classification function for own image

if __name__ == '__main__':
    main()
