
# coding: utf-8

import os

import cv2
import numpy as np
from tqdm import tqdm
import dlib

from config import IMG_SIZE
from models.mobile_net import MobileNetDeepEstimator
from preprocessor import preprocess_input

detector = dlib.get_frontal_face_detector()


def preprocess(image_arr):
    data = preprocess_input(image_arr)
    return data


def detect_faces(img):
    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detected = detector(input_img, 1)
    faces = np.empty((len(detected), IMG_SIZE, IMG_SIZE, 3))
    img_h, img_w, _ = np.shape(input_img)

    for i, d in tqdm(enumerate(detected)):
        x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
        xw1 = max(int(x1 - 0.4 * w), 0)
        yw1 = max(int(y1 - 0.4 * h), 0)
        xw2 = min(int(x2 + 0.4 * w), img_w - 1)
        yw2 = min(int(y2 + 0.4 * h), img_h - 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (IMG_SIZE, IMG_SIZE))

    return faces


model = MobileNetDeepEstimator(IMG_SIZE, 1, 21, weights=None)()
model.load_weights(os.path.join("checkpoints", "weights.62-1.97.hdf5"))

base_path = "test"

with open('prediction.csv', 'w') as f:
    f.write('Age, Gender')

with open("prediction.csv", "a") as f:
    for _, _, imgs in os.walk(base_path):
        for im in tqdm(imgs):
            img = cv2.imread(os.path.join(base_path, im))

            img_data = detect_faces(img)
            img_data = preprocess(img_data)

            results = model.predict(img_data)
            predicted_gender = results[0]
            ages = np.arange(0, 21).reshape(21, 1)
            predicted_age = results[1].dot(ages).flatten()
            res = '{},{}\n'.format(im,
                                   int(predicted_age[0]*4.76),
                                   predicted_gender[0])
            print(res)
            f.write(res)

