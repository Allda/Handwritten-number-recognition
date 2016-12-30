########################################################
# Authors: Ales Raszka, Marek Fiala, Matus Dobrotka
# Project: POV
# Year: 2016
########################################################

import sys

import argparse
import numpy as np

from mnist import MNIST
from opencvClassifiers import openCVClassifier
from ownNumberProcessor import OwnNumberProcessor
from sklearnClassifiers import sklearnClassifier

from constants import classifiers

def setup_parser():
    """Function that create parser object that parse command line parameters"""
    parser = argparse.ArgumentParser(description='Hand writen number '
                                                 ' recognition')
    main_group = parser.add_mutually_exclusive_group()
    main_group.add_argument('--train-sklearn-classifier', default=None,
                            metavar='CLASSIFIER',
                            choices=classifiers,
                            help='Train classifier. Choices: %s' %
                                 ', '.join(classifiers))
    main_group.add_argument('--train-opencv-classifier', default=None,
                            metavar='CLASSIFIER',
                            choices=classifiers,
                            help='Train classifier. Choices: %s' %
                                 ', '.join(classifiers))
    main_group.add_argument('--classify-mnist-opencv', action='store_true',
                            help='Classify MNIST database by opencv classifier')
    main_group.add_argument('--classify-own-opencv', default=None, metavar='FILE',
                            help='Clasify own picture with handwritten '
                                 'numbers by opencv classifier')
    main_group.add_argument('--classify-mnist-sklearn', action='store_true',
                            help='Classify MNIST database by sklearn classifier')
    main_group.add_argument('--classify-own-sklearn', default=None, metavar='FILE',
                            help='Clasify own picture with handwritten '
                                 'numbers by sklearn classifier')

    parser.add_argument('--mnist', default=None, nargs='?',
                        help='Location of MNIST database. In case it is '
                             'missing, it will try to download database')

    parser.add_argument('--block-size', default=None, metavar='Tuple', nargs='+', type=int,
                            help='Specify size of block - for example 7 7')
    parser.add_argument('--cell-size', default=None, metavar='Tuple', nargs='+', type=int,
                            help='Specify size of cell - for example 7 7')
    parser.add_argument('--bins', default=None, metavar='int',
                            help='Specify count of the bins.')

    return parser


def main():
    """Main function that process actions"""
    parser = setup_parser()
    args = parser.parse_args()

    mnist = MNIST()

    block_size = (14, 14)
    cell_size = (7, 7)
    nbins = 9

    # Parameters that specify HOG computation
    if args.block_size:
        block_size = tuple(args.block_size)

    if args.cell_size:
        cell_size = tuple(args.cell_size)

    if args.bins:
        nbins = int(args.bins)

    if args.train_sklearn_classifier:
        classifier = sklearnClassifier()

        classifier.init_hog(block_size=block_size, cell_size=cell_size, nbins=nbins)
        classifier.create_classifier(args.train_sklearn_classifier)

        print 'train sklearn %s' % args.train_sklearn_classifier
        training_imgs, training_labels = mnist.load_training()
        training_imgs = np.array(training_imgs)

        classifier.train_classifier(training_imgs, training_labels)
        classifier.save_classifier()

        testing_imgs, testing_labels = mnist.load_testing()
        testing_imgs = np.array(testing_imgs)
        testing_labels = np.array(testing_labels)
        classifier.test_classifier(testing_imgs, testing_labels)

    elif args.train_opencv_classifier:
        classifier = openCVClassifier()
        classifier.init_hog(block_size=block_size, cell_size=cell_size, nbins=nbins)
        classifier.create_classifier(args.train_opencv_classifier)

        print 'train opencv %s' % args.train_opencv_classifier
        training_imgs, training_labels = mnist.load_training()
        training_imgs = np.array(training_imgs)
        training_labels = np.array(training_labels)

        classifier.train_classifier(training_imgs, training_labels)
        classifier.save_classifier()

        testing_imgs, testing_labels = mnist.load_testing()
        testing_imgs = np.array(testing_imgs)
        testing_labels = np.array(testing_labels)
        classifier.test_classifier(testing_imgs, testing_labels)

    elif args.classify_mnist_sklearn:
        classifier = sklearnClassifier()
        classifier.init_hog(block_size=block_size, cell_size=cell_size, nbins=nbins)
        classifier.load_classifier()

        testing_imgs, testing_labels = mnist.load_testing()
        testing_imgs = np.array(testing_imgs)
        classifier.test_classifier(testing_imgs, testing_labels)

    elif args.classify_mnist_opencv:
        print "OpenCV in current version do not support loading of classifier"
        sys.exit(2)

        classifier = openCVClassifier()
        # classifier.load_classifier()

        testing_imgs, testing_labels = mnist.load_testing()
        testing_imgs = np.array(testing_imgs)
        classifier.test_classifier(testing_imgs, testing_labels)

    elif args.classify_own_sklearn:
        print 'classify own picture %s' % args.classify_own_opencv
        classifier = sklearnClassifier()
        classifier.init_hog(block_size=block_size, cell_size=cell_size, nbins=nbins)
        classifier.load_classifier()
        processor = OwnNumberProcessor()
        processor.set_img(args.classify_own_sklearn)
        processor.process(classifier)

    elif args.classify_own_opencv:
        print "OpenCV in current version do not support loading of classifier"
        sys.exit(2)

        classifier = openCVClassifier()
        # classifier.load_classifier()

        processor = OwnNumberProcessor()
        processor.set_img(args.classify_own_opencv)
        processor.process(classifier)

if __name__ == '__main__':
    main()
