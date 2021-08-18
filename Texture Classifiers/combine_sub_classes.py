import os
import shutil
import cv2
import numpy as np


def is_similar(image1, image2):
    return image1.shape == image2.shape and not (np.bitwise_xor(image1, image2).any())


dir_ = 'top'
for label in os.listdir(dir_):
    label_dir = os.path.join(dir_, label)
    for sub_label in os.listdir(label_dir):
        sub_label_dir = os.path.join(label_dir, sub_label)
        if not os.path.isdir(sub_label_dir):
            continue
        for img_name in os.listdir(sub_label_dir):
            img_path = os.path.join(sub_label_dir, img_name)
            try:
                shutil.move(img_path, label_dir)
            except shutil.Error:
                img = cv2.imread(img_path)
                target_img = cv2.imread(os.path.join(label_dir, img_name))
                if is_similar(img, target_img):
                    os.remove(img_path)
                else:
                    shutil.move(img_path, os.path.join(label_dir, "duplicate.jpg"))

        shutil.rmtree(sub_label_dir)
# shutil.rmtree(dir_)
