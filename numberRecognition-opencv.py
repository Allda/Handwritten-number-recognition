 
import numpy as np
import cv2
from mnist import MNIST

mnist = MNIST()
training_imgs, training_labels = mnist.load_training()
training_imgs = np.array(training_imgs).astype(np.float32)
training_labels = np.array(training_labels)

testing_imgs, testing_labels = mnist.load_testing()
testing_imgs = np.array(testing_imgs).astype(np.float32)
testing_labels = np.array(testing_labels)

# K-Nearest classifier
knn = cv2.ml.KNearest_create()
knn.train(training_imgs,cv2.ml.ROW_SAMPLE,training_labels)
ret,result,neighbours,dist = knn.findNearest(testing_imgs,k=5)

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