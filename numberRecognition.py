import sys
import json
import argparse
import numpy as np

from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans

from sklearn.externals import joblib

from mnist import MNIST
from ownNumberProcessor import OwnNumberProcessor

classifier_file_name = 'classifier.pkl'

LINEAR = "linear"
K_MEANS = "k-means"
K_NEAREST = "k-nearest"
ADABOOST = "adaboost"

def setup_parser():
    parser = argparse.ArgumentParser(description='Hand writen number '
                                                 ' recognition')
    main_group = parser.add_mutually_exclusive_group()
    main_group.add_argument('--train-classifier', default=None, metavar='CLASSIFIER',
                            help='Train classifier- possible classifiers linear, k-nearest, k-means, adaboost')
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


def train_classifier(images, labels, classifier):
    image_hog_list = []
    for image in images:
        image_hog = hog(image.reshape((28, 28)), orientations=9,
                        pixels_per_cell=(14, 14), cells_per_block=(1, 1),
                        visualise=False)
        image_hog_list.append(image_hog)
    hog_features = np.array(image_hog_list, 'float64')

    classifier.fit(hog_features, labels)

    return classifier


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

    if args.train_classifier:
        if args.train_classifier == LINEAR:
            classifier = LinearSVC()
        elif args.train_classifier == K_NEAREST:
            classifier = KNeighborsClassifier()
        elif args.train_classifier == ADABOOST:
            classifier = AdaBoostClassifier(DecisionTreeClassifier(),
                           algorithm="SAMME",
                           n_estimators=200)
        elif args.train_classifier == K_MEANS:
            classifier = KMeans(n_clusters=10) # We have ten numbers
        else:
            print "No correct classifier set: %s" % args.train_classifier
            sys.exit(-1)

        print 'train %s' % args.train_classifier
        training_imgs, training_labels = mnist.load_training()
        training_imgs = np.array(training_imgs)
        classifier = train_classifier(training_imgs, training_labels, classifier)
        joblib.dump(classifier, classifier_file_name, compress=3)
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

if __name__ == '__main__':
    main()
