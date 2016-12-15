import argparse
import numpy as np

from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

from mnist import MNIST


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

    return parser


def train_classifier(images, labels):
    image_hog_list = []
    for image in images:
        image_hog = hog(image.reshape((28, 28)), orientations=9,
                        pixels_per_cell=(14, 14), cells_per_block=(1, 1),
                        visualise=False)
        image_hog_list.append(image_hog)
    hog_features = np.array(image_hog_list, 'float64')

    lcsv = LinearSVC()
    lcsv.fit(hog_features, labels)

    return lcsv


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

    mnist = MNIST()
    img, labels = mnist.load_training()
    img = np.array(img)

    for i in range(50):
        print MNIST.display(img[i]), labels[i]
    if args.train_classifier:
        print 'train'
        lcsv = train_classifier(img, labels)
        joblib.dump(lcsv, 'classifier.pkl', compress=3)
    elif args.classify_mnist:
        print 'clasify mnist'
        lcvs = joblib.load("classifier.pkl")
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
