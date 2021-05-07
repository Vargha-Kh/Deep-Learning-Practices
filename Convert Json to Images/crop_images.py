import os
import math
import cv2
import json


def cropping_box(dataset_dir, json_path, output_path, input_path):
    annotations = json.load(open(os.path.join(dataset_dir, json_path)))
    annotations = list(annotations.values())
    image_file = annotations[1]
    annotations_point = annotations[2]
    for annotations_object in annotations_point:
        file_name = image_file[annotations_object['image_id'] - 1]['file_name']
        bbox = annotations_object['bbox']
        x, y, width, height = bbox[0], bbox[1], bbox[2], bbox[3]
        x = math.ceil(x)
        y = math.ceil(y)
        width = math.ceil(width)
        height = math.ceil(height)
        img = os.path.join(input_path, file_name)
        img = cv2.imread(img)
        crop_img = img[y:y + height, x:x + width]
        cv2.imwrite(os.path.join(output_path, file_name), crop_img)


dataset_directory = "/home/vargha/Desktop"
json_file_path = "VARGHA.58.FCSEG.json"
output_dir_path = "/home/vargha/Desktop/Export"
input_dir_path = "/home/vargha/Desktop/images"
cropping_box(dataset_directory, json_file_path, output_dir_path, input_dir_path)
