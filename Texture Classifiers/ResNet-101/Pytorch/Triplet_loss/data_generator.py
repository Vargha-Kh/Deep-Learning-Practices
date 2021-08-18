import os
from collections import defaultdict
from datetime import datetime
from random import choice
import cv2
import torch.utils.data
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np


def get_label_list(img_dir, base_path, valid=False):
    if valid:
        group = 'valid'
    else:
        group = 'train'
    img_dir = os.path.join(base_path, img_dir, group)
    map_ = {class_name: en for en, class_name in enumerate(os.listdir(img_dir))}
    labels = []
    img_list = []
    for class_name in os.listdir(img_dir):
        class_dir = os.path.join(img_dir, class_name)
        label = map_[class_name]
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img_list.append(img_path)
            labels.append(label)
    return img_list, labels


class DataGenerator(Dataset):

    def __init__(self, img_dir, base_dir, batch_size=1, dim=(224, 224), n_channels=3,
                 n_classes=7, shuffle=True, image_dir='.', valid=False):
        self.dim = dim
        self.image_dir = image_dir
        self.batch_size = batch_size
        img_ids, labels = get_label_list(img_dir, base_dir, valid)
        self.img_ids = img_ids
        self.labels = labels
        self.class_labels = defaultdict(list)
        for label, img_path in zip(self.labels, self.img_ids):
            self.class_labels[label].append(img_path)
        self.label_names = list(self.class_labels.keys())
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.indexes = None
        self.on_epoch_end()
        self.val = valid

    def __len__(self):
        return int(np.floor(len(self.img_ids) / self.batch_size))

    def __getitem__(self, index):
        # indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # list_IDs_temp = [self.img_ids[k] for k in indexes]
        # labels = [self.labels[k] for k in indexes]
        list_IDs_temp = self.img_ids
        labels = self.labels
        X, y = self.__data_generation(list_IDs_temp, labels)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.img_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp, labels):
        X_anchor = np.zeros((self.n_channels, *self.dim), dtype=float)
        X_neg = np.zeros((self.n_channels, *self.dim), dtype=float)
        X_pos = np.zeros((self.n_channels, *self.dim), dtype=float)
        y = np.zeros(1, dtype=float)
        for i, (name, label) in enumerate(zip(list_IDs_temp, labels)):
            img = self.get_image(name)
            positive_path = choice(self.class_labels[label])
            pos_img = self.get_image(positive_path)
            labs = self.label_names.copy()
            labs.remove(label)
            neg_label = choice(labs)
            neg_path = choice(self.class_labels[neg_label])
            neg_img = self.get_image(neg_path)
            X_anchor = img
            X_neg = neg_img
            X_pos = pos_img

        return [X_anchor, X_pos, X_neg], y

    def get_image(self, name):
        img = cv2.imread(name)[..., ::-1]
        # img = preprocess_input(img)
        img = self.resize_pad(img, self.dim[0])
        img = img.transpose((2, 0, 1))
        return img

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
