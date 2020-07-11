# *_*coding:utf-8 *_*
#
import os

import cv2

data_dir = 'D:\\pycharm_workspace\\Facenet-Real-time-face-recognition-using-deep-learning-Tensorflow\\train_img\\CTeacher'

for root, dirs, files in os.walk(data_dir):
    for file in files:
        path = os.path.join(root, file)
        img = cv2.imread(path)
        img = cv2.resize(img, (800, 1000), 0.5, 0.5)
        cv2.imwrite(path,img)
        # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
