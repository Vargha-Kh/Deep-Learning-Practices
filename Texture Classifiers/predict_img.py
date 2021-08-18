import time
import os
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from resnet101V2.model import get_model


def resize_pad(im, desired_size):
    old_size = im.shape[:2]  # old_size is in (height, width) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return new_im


dir_ = "/home/symo-pouya/projects/texture/test/crop"
size = (512, 512)
weight_path = 'resnet101V2/resnet101_v2_fine.h5'
model = get_model(7, weights=weight_path, dim=(*size, 3), fine_tune=True, predict=True)
class_names = {'zigzag': 0,
               'picture': 1,
               'plaid': 2,
               'stripe': 3,
               'plain': 4,
               'hawaei': 5,
               'spotted': 6}
class_names = {v: k for k, v in class_names.items()}
print(model.summary())
img_path = "top/zigzag/1 (1).jpg"
img = cv2.imread(img_path)[..., ::-1]
img = resize_pad(img, size[0])
images = np.expand_dims(img, axis=0)
model.predict(images)[0]
tic = time.time()
prediction = model.predict(images)[0]
toc = time.time()
print('prediction time:', toc - tic)
pred = int(np.argmax(prediction))
p = sorted([(j, class_names[i]) for i, j in enumerate(prediction)], key=lambda x: x[0], reverse=True)
print(p)
cv2.imshow('', img[..., ::-1])
cv2.waitKey(0)
