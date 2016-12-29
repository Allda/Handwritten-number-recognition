import numpy as np
import cv2
from mnist import MNIST

classifier_file_name = 'opencv_SVM_classifier.xml'

win_size = (28, 28)
nbins = 9
block_size = (14, 14)
block_stride = (14, 14)
cell_size = (14, 14)

hog_desc = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

svm_params = dict( kernel_type = cv2.ml.SVM_LINEAR,
                    svm_type = cv2.ml.SVM_C_SVC,
                    C=2.67, gamma=5.383 )

affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

def train_classifier(images, labels):
    image_hog_list = []
    
    for image in images:
        image1 = np.uint8(image.reshape((28, 28)))
        image_hog = hog_desc.compute(image1)
        image_hog_list.append(image_hog)
        
    hog_features = np.array(image_hog_list, 'float32')

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setGamma(5.383)
    svm.setC(2.67)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.train(hog_features, cv2.ml.ROW_SAMPLE, np.array(labels, dtype = np.int32))
    return svm
    
def test_classifier(testing_imgs, testing_labels, svm):
    image_hog_list_test = []
    for image in testing_imgs:
        image1 = np.uint8(image.reshape((28, 28)))

        image_hog = hog_desc.compute(image1)
        image_hog_list_test.append(image_hog)

    hog_features_test = np.array(image_hog_list_test, 'float32')

    result = svm.predict(hog_features_test)

    i = 0
    ok = 0
    fail = 0
    for row in result[1]:
        if row == testing_labels[i]:
            ok += 1
        else:
            fail += 1
        i += 1

    accuracy = ok*100.0/result[1].size

    return accuracy


def main():
    mnist = MNIST()
    training_imgs, training_labels = mnist.load_training()
    training_imgs = np.array(training_imgs)
    testing_imgs, testing_labels = mnist.load_testing()
    testing_imgs = np.array(testing_imgs)

    svm = train_classifier(training_imgs, training_labels)
    print "Classifer was saved to file: %s" % classifier_file_name
    svm.save(classifier_file_name)

    accuracy = test_classifier(testing_imgs, testing_labels, svm)
    print accuracy	

if __name__ == '__main__':
    main()