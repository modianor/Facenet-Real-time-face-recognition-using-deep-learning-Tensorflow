# -*- coding: utf-8 -*-
"""
识别图像的类，为了快速进行多次识别可以调用此类下面的方法：
R = Recognizer(image_height, image_width, max_captcha)
for i in range(10):
    r_img = Image.open(str(i) + ".jpg")
    t = R.rec_image(r_img)
简单的图片每张基本上可以达到毫秒级的识别速度
"""
import json
import os
import pickle

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy import misc

import detect_face
import facenet

img_path = 'abc.jpg'
modeldir = './model/20170511-185253.pb'
classifier_filename = './class/classifier.pkl'
npy = './npy'
train_img = "./train_img"


class Recognizer(object):
    def __init__(self, img_path='abc.jpg', modeldir='./model/20170511-185253.pb',
                 classifier_filename='./class/classifier.pkl', npy='./npy', train_img="./train_img"):
        # 初始化变量
        self.img_path = img_path
        self.modeldir = modeldir
        self.classifier_filename = classifier_filename
        self.train_img = train_img
        self.minsize = 20  # minimum size of face
        self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        self.factor = 0.709  # scale factor
        self.margin = 44
        self.frame_interval = 3
        self.batch_size = 1000
        self.image_size = 182
        self.input_image_size = 160
        self.HumanNames = os.listdir(self.train_img)
        self.HumanNames.sort()
        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g,
                               config=tf.ConfigProto(gpu_options=self.gpu_options, log_device_placement=False))
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)

        with self.sess.as_default():
            self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, npy)
            print('Loading feature extraction model')
            self.facenet = facenet.load_model(self.modeldir)
            self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            self.embedding_size = self.embeddings.get_shape()[1]

            classifier_filename_exp = os.path.expanduser(self.classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (self.model, self.class_names) = pickle.load(infile)

    def rec_image(self, frame):

        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # resize frame (optional)

        find_results = []

        if frame.ndim == 2:
            frame = self.facenet.to_rgb(frame)
        frame = frame[:, :, 0:3]
        bounding_boxes, _ = detect_face.detect_face(frame, self.minsize, self.pnet, self.rnet, self.onet,
                                                    self.threshold, self.factor)
        nrof_faces = bounding_boxes.shape[0]
        print('Face Detected: %d' % nrof_faces)

        if nrof_faces > 0:
            det = bounding_boxes[:, 0:4]
            img_size = np.asarray(frame.shape)[0:2]

            cropped = []
            scaled = []
            scaled_reshape = []
            bb = np.zeros((nrof_faces, 4), dtype=np.int32)

            for i in range(nrof_faces):
                emb_array = np.zeros((1, self.embedding_size))

                bb[i][0] = det[i][0]
                bb[i][1] = det[i][1]
                bb[i][2] = det[i][2]
                bb[i][3] = det[i][3]

                # inner exception
                if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                    print('face is too close')
                    continue

                cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                cropped[i] = facenet.flip(cropped[i], False)
                scaled.append(misc.imresize(cropped[i], (self.image_size, self.image_size), interp='bilinear'))
                scaled[i] = cv2.resize(scaled[i], (self.input_image_size, self.input_image_size),
                                       interpolation=cv2.INTER_CUBIC)
                scaled[i] = facenet.prewhiten(scaled[i])
                scaled_reshape.append(scaled[i].reshape(-1, self.input_image_size, self.input_image_size, 3))
                feed_dict = {self.images_placeholder: scaled_reshape[i], self.phase_train_placeholder: False}
                emb_array[0, :] = self.sess.run(self.embeddings, feed_dict=feed_dict)
                predictions = self.model.predict_proba(emb_array)
                print(predictions)
                best_class_indices = np.argmax(predictions, axis=1)
                # print(best_class_indices)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                print(best_class_probabilities)
                cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)  # boxing face

                # plot result idx under box
                text_x = bb[i][0]
                text_y = bb[i][3] + 20
                print('Result Indices: ', best_class_indices[0])
                print(self.HumanNames)
                for H_i in self.HumanNames:
                    # print(H_i)
                    if self.HumanNames[best_class_indices[0]] == H_i:
                        result_names = self.HumanNames[best_class_indices[0]]
                        cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (0, 0, 255), thickness=1, lineType=2)

        # 返回识别结果
        return ''


def main():
    with open("conf/sample_config.json", "r", encoding="utf-8") as f:
        sample_conf = json.load(f)
    image_height = sample_conf["image_height"]
    image_width = sample_conf["image_width"]
    max_captcha = sample_conf["max_captcha"]
    char_set = sample_conf["char_set"]
    model_save_dir = sample_conf["model_save_dir"]
    R = Recognizer(image_height, image_width, max_captcha, char_set, model_save_dir)
    r_img = Image.open("./sample/test/2b3n_6915e26c67a52bc0e4e13d216eb62b37.jpg")
    t = R.rec_image(r_img)
    print(t)


if __name__ == '__main__':
    main()
