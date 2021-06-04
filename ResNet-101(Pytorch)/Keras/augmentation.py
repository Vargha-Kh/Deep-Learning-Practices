import os
from random import choice
import tensorflow as tf
import numpy as np
import cv2
import imgaug.augmenters as iaa


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, img_dir, base_dir, batch_size=32, dim=(224, 224), n_channels=3, augment=True,
                 shuffle=True, image_dir='.', val=False, aug_prop=0.7, return_path=False):
        self.return_path = return_path
        self.dim = dim
        self.augment = augment
        self.aug_prop = aug_prop
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.img_ids, self.labels, self.class_name_map = self.get_label_list(img_dir, base_dir)
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.val = val
        # self.on_epoch_end()
        self.indexes = None
        self.on_epoch_end()

    def aug(self, label, h, w):

        seq_1 = iaa.Sequential([
            iaa.Fliplr(1.0),
            iaa.Rotate(rotate=(-30, 30)),
            iaa.ScaleX(scale=(0.8, 1.0)),
            iaa.MultiplyHue((0.5, 1.5)),
        ])

        seq_2 = iaa.Sequential([
            iaa.PiecewiseAffine(scale=0.05),
            iaa.Rotate((-30, 30)),
            iaa.TranslateX(px=(-30, 30)),
            iaa.TranslateY(px=(-30, 30)),
            iaa.Grayscale(0.6)
        ])

        seq_3 = iaa.Sequential([
            iaa.Crop(percent=0.1),
            iaa.ScaleX(scale=(0.8, 0.9)),
            iaa.Fliplr(0.5),
            iaa.ChangeColorTemperature((1100, 10000)),
            iaa.MultiplyHue((0.5, 1.5))
        ])

        seq_4 = iaa.Sequential([
            iaa.Affine((1.1, 1.3)),
            iaa.Rotate((-5, 5)),
            iaa.TranslateX(percent=0.04),
            iaa.ShearX((-15, 15))
        ])

        seq_5 = iaa.Sequential([
            iaa.Fliplr(1.0),
            iaa.Crop(percent=0.02),
            iaa.TranslateX(percent=[-0.1, 0.1]),
            iaa.ChangeColorTemperature((6000, 11000))
        ])

        seq_6 = iaa.Sequential([
            iaa.PiecewiseAffine(scale=0.05),
            iaa.Rotate((-30, 30)),
            iaa.TranslateX(px=(-30, 30)),
            iaa.TranslateY(px=(-30, 30)),
        ])

        aug_dict = {
            'hawaei': [seq_1, seq_2, seq_3, seq_4, seq_5],
            'picture': [seq_1, seq_2, seq_3, seq_4, seq_5],
            'plaid': [seq_1, seq_3, seq_4, seq_5],
            'plain': [seq_1, seq_2, seq_3, seq_4, seq_5],
            'spotted': [seq_1, seq_3, seq_4, seq_5, seq_6],
            'stripe': [seq_1, seq_3, seq_4, seq_5],
            'zigzag': [seq_1, seq_2, seq_3, seq_4, seq_5],

        }
        seq = choice(aug_dict[label])
        return iaa.Sometimes(self.aug_prop, seq)

    @staticmethod
    def get_label_list(img_dir, base_path):
        img_dir = os.path.join(base_path, img_dir)
        class_name_map = {class_name: en for en, class_name in enumerate(os.listdir(img_dir))}
        labels = []
        img_list = []
        for class_name in os.listdir(img_dir):
            class_dir = os.path.join(img_dir, class_name)
            label = class_name
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img_list.append(img_path)
                labels.append(label)
        return img_list, labels, class_name_map

    def __len__(self):
        return int(np.floor(len(self.img_ids) / self.batch_size))

    @staticmethod
    def resize_pad(im, desired_size=224):
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
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=color)

        return new_im

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.img_ids[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]
        x, y = self.__data_generation(list_IDs_temp, labels)
        if self.return_path:
            return x, y, list_IDs_temp
        else:
            return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.img_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp, labels):
        x = np.zeros((self.batch_size, *self.dim, self.n_channels))
        y = np.zeros(self.batch_size, dtype=float)
        for i, (name, label) in enumerate(zip(list_IDs_temp, labels)):
            img = self.get_image(name)
            h, w, _ = img.shape
            if not self.val and self.augment:
                aug = self.aug(label, h=h, w=w)
                img = aug(image=img)
            x[i] = img
            y[i] = self.class_name_map[label]

        return x, y

    def get_image(self, name):
        img = cv2.imread(name)[..., ::-1]
        img = self.resize_pad(img, self.dim[0])
        return img
