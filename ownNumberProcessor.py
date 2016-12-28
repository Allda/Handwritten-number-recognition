import numpy as np
from skimage.feature import hog

import cv2


class OwnNumberProcessor(object):

    def __init__(self):
        self.img = None

    def set_img(self, img_name):
        self.img = cv2.imread(img_name)

    def process(self, lcvs):

        if not len(self.img[0:0]) == 1:
            gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = self.img

        gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

        ret, process_img = cv2.threshold(gray_img, 90, 255, cv2.THRESH_BINARY_INV)
        _, ctrs, _ = cv2.findContours(process_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rects = [cv2.boundingRect(ctr) for ctr in ctrs]

        for rect in rects:
            cv2.rectangle(self.img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)

            leng = int(rect[3] * 2)
            pt1 = int(rect[1] + rect[3] / 2 - leng / 2)
            pt2 = int(rect[0] + rect[2] / 2 - leng / 2)

            if pt1 < 0:
                pt1 = 0

            if pt2 < 0:
                pt2 = 0

            roi = process_img[pt1:pt1 + leng, pt2:pt2 + leng]
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi = cv2.dilate(roi, (3, 3))

            roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
            nbr = lcvs.predict(np.array([roi_hog_fd], 'float64'))
            cv2.putText(self.img, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

        cv2.imshow("Rectangles", self.img)
        cv2.waitKey()