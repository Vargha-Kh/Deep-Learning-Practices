import os
import numpy as np
import math
import cv2
import json
from PIL import Image, ImageDraw


def cropping_segment(dataset_dir, json_path, output_path, input_path):
    annotations = json.load(open(os.path.join(dataset_dir, json_path)))
    annotations = list(annotations.values())
    image_file = annotations[1]
    annotations_point = annotations[2]
    for box in annotations_point:
        file_name = image_file[box['image_id'] - 1]['file_name']
        segmentation = box['segmentation'][0]
        seg = []
        for i in range(0, len(segmentation), 2):
            point = (segmentation[i], segmentation[i + 1])
            seg.append(point)
        original = Image.open(os.path.join(input_path, file_name)).convert("RGBA")
        mask = Image.new("L", original.size, 0)
        ImageDraw.Draw(mask).polygon(seg, fill=255, outline=1)
        copy = Image.new("RGBA", original.size, 255)
        result = Image.composite(original, copy, mask)
        result.save(os.path.join(output_path, file_name.split('.')[0] + '.png'))


dataset_directory = "/home/vargha/Desktop"
json_file_path = "VARGHA.58.FCSEG.json"
output_dir_path = "/home/vargha/Desktop/Export"
input_dir_path = "/home/vargha/Desktop/images"
cropping_segment(dataset_directory, json_file_path, output_dir_path, input_dir_path)
