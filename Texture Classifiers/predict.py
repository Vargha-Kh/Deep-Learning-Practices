import os
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from vgg16.model import get_model


def predict(name, dir_, size, resize=True, pad=True, org=True):
    if org:
        org_img = cv2.imread(os.path.join(dir_ + "/../", name))[..., ::-1]
    else:
        org_img = cv2.imread(os.path.join(dir_, name))[..., ::-1]
    if resize:
        if pad:
            org_img = resize_pad(org_img, size[0])
        else:
            org_img = cv2.resize(org_img, size)
    images = np.expand_dims(org_img, axis=0)
    prediction = model.predict(images)[0]
    pred = int(np.argmax(prediction))
    # print(name, class_names[pred])
    p = sorted([(j, class_names[i]) for i, j in enumerate(prediction)], key=lambda x: x[0], reverse=True)
    print(p)
    cv2.imshow('', org_img[..., ::-1])
    cv2.waitKey(0)


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
size = (224, 224)
weight_path = '/home/symo-pouya/projects/texture/vgg16/vgg16_fine.h5'
model = get_model(7, weights=weight_path, dim=(*size, 3), fine_tune=True, predict=True)
class_names = {'plaid': 0,
               'spotted': 1,
               'hawaei': 2,
               'stripe': 3,
               'plain': 4,
               'picture': 5,
               'zigzag': 6}
class_names = {v: k for k, v in class_names.items()}
print(model.summary())

for name in os.listdir(dir_):
    predict(name, dir_, size,org=True, pad= )
    # predict(name, dir_, size, org=True, pad=False)
