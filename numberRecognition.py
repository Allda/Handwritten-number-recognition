import sys

import argparse
import numpy as np


from sklearn.externals import joblib

from mnist import MNIST
from opencvClassifiers import openCVClassifier
from ownNumberProcessor import OwnNumberProcessor
from sklearnClassifiers import sklearnClassifier

classifiers = [LINEAR, K_MEANS, K_NEAREST, ADABOOST]
def setup_parser():
    parser = argparse.ArgumentParser(description='Hand writen number '
                                                 ' recognition')
    main_group = parser.add_mutually_exclusive_group()
    main_group.add_argument('--train-classifier', default=None,
                            metavar='CLASSIFIER',
                            choices=classifiers,
                            help='Train classifier. Choices: %s' %
                                 ', '.join(classifiers))
    main_group.add_argument('--train-opencv-classifier', default=None, metavar='CLASSIFIER',
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


def main():
    parser = setup_parser()
    args = parser.parse_args()
    if args.classifier_file:
        classifier_file_name = args.classifier_file

    mnist = MNIST()

    if args.train_sklearn_classifier:
        classifier = sklearnClassifier()
        classifier.create_classifier(args.train_sklearn_classifier)

        print 'train sklearn %s' % args.train_sklearn_classifier
        training_imgs, training_labels = mnist.load_training()
        training_imgs = np.array(training_imgs)

        classifier.train_classifier(training_imgs, training_labels)
        classifier.save_classifier()

    elif args.train_opencv_classifier:
        classifier = openCVClassifier()
        classifier.create_hog()
        classifier.create_classifier(args.train_opencv_classifier)

        print 'train opencv %s' % args.train_opencv_classifier
        training_imgs, training_labels = mnist.load_training()
        training_imgs = np.array(training_imgs)
        training_labels = np.array(training_labels)

        classifier.train_classifier(training_imgs, training_labels)

        testing_imgs, testing_labels = mnist.load_testing()
        testing_imgs = np.array(testing_imgs)
        testing_labels = np.array(testing_labels)
        classifier.test_classifier(testing_imgs, testing_labels)
        # classifier.save_classifier()

    elif args.classify_mnist:
        print 'clasify mnist'
        classifier = sklearnClassifier()
        classifier.load_classifier()

        testing_imgs, testing_labels = mnist.load_testing()
        testing_imgs = np.array(testing_imgs)
        classifier.test_classifier(testing_imgs, testing_labels)

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
