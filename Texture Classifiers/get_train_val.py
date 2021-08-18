import os
import shutil
from collections import defaultdict
from sklearn.model_selection import train_test_split

img_dir = 'top'
base_path = '.'
percent = 0.1


def get_label_list(img_dir, base_path):
    img_dir = os.path.join(base_path, img_dir)
    label_list = defaultdict(list)
    for class_name in os.listdir(img_dir):
        class_dir = os.path.join(img_dir, class_name)
        label = class_name
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            label_list[label].append(img_path)
    return label_list


label_list = get_label_list(img_dir, base_path)
for label, images in label_list.items():
    train, test = train_test_split(images, test_size=percent)
    label_path = os.path.join(base_path, 'validation', img_dir, label)
    os.makedirs(label_path, exist_ok=True)
    for img_path in test:
        shutil.move(img_path, label_path)
