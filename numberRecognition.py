import sys
import argparse
import numpy as np

from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

from mnist import MNIST

classifier_file_name = 'classifier.pkl'

def setup_parser():
    parser = argparse.ArgumentParser(description='Hand writen number '
                                                 ' recognition')
    main_group = parser.add_mutually_exclusive_group()
    main_group.add_argument('--train-classifier', action='store_true',
                            help='Train classifier')
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


def train_classifier(images, labels):
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


def classify_MNIST_data(images, labels, lcvs):
    result = []
    for index, image in enumerate(images):
        image_hog = hog(image.reshape((28, 28)), orientations=9,
                        pixels_per_cell=(14, 14), cells_per_block=(1, 1),
                        visualise=False)
        predicted_value = lcvs.predict(np.array([image_hog]))[0]
        result.append((labels[index], predicted_value))
    return result


def main():
    parser = setup_parser()
    args = parser.parse_args()
    if args.classifier_file:
        global classifier_file_name
        classifier_file_name = args.classifier_file

    mnist = MNIST()

    if args.train_classifier:
        print 'train'
        training_imgs, training_labels = mnist.load_training()
        training_imgs = np.array(training_imgs)
        lcvs = train_classifier(training_imgs, training_labels)
        joblib.dump(lcvs, classifier_file_name, compress=3)
        print "Classifer was save to file: %s" % classifier_file_name
    elif args.classify_mnist:
        print 'clasify mnist'
        try:
            lcvs = joblib.load(classifier_file_name)
        except IOError:
            print "No classifier file has been found: %s" % classifier_file_name
            sys.exit(-1)
        testing_imgs, testing_labels = mnist.load_testing()
        testing_imgs = np.array(testing_imgs)
        result = classify_MNIST_data(testing_imgs, testing_labels, lcvs)
        correct = 0
        for item in result:
            if item[0] == item[1]:
                correct += 1
        print "%s %s %s" % (correct, len(result),
                            float(correct)/len(result) * 100)
        # TODO: call classification function for MNIST data
    elif args.classify_own:
        print 'classify own picture'
        print args.classify_own
        # TODO: call classification function for own image

if __name__ == '__main__':
    main()
