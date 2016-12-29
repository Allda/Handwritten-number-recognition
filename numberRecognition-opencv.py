 
import numpy as np
import cv2
from mnist import MNIST
from skimage.feature import hog

mnist = MNIST()
training_imgs, training_labels = mnist.load_training()
training_imgs = np.array(training_imgs)
training_labels = np.array(training_labels)

testing_imgs, testing_labels = mnist.load_testing()
testing_imgs = np.array(testing_imgs)
testing_labels = np.array(testing_labels)


win_size = (28, 28)
nbins = 9 # orientations in sklearn
block_size = (14, 14)
block_stride = (14, 14)
cell_size = (14, 14) # pixels_per_cell


hog2 = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
image_hog_list = []
for image in training_imgs:
    vis2 = np.uint8(image.reshape((28, 28)))

    image_hog = hog2.compute(vis2)
    image_hog_list.append(image_hog)

hog_features = np.array(image_hog_list, 'float32')

image_hog_list_test = []
for image in testing_imgs:
    vis2 = np.uint8(image.reshape((28, 28)))

    image_hog = hog2.compute(vis2)
    image_hog_list_test.append(image_hog)

hog_features_test = np.array(image_hog_list_test, 'float32')

# K-Nearest classifier
knn = cv2.ml.KNearest_create()
# knn.load("test.xml")

knn.train(hog_features, cv2.ml.ROW_SAMPLE, training_labels)
ret,result,neighbours,dist = knn.findNearest(hog_features_test, k=5)

knn.save("test.xml")

i = 0
ok = 0
fail = 0
for row in result:
    if row == testing_labels[i]:
        ok += 1
    else:
        fail += 1
    i += 1

accuracy = ok*100.0/result.size
print accuracy