import os.path
from collections import namedtuple
import cv2
import json
import math


def intersection_over_union(box_a, box_b):
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    intersection_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    return intersection_area / float(box_a_area + box_b_area - intersection_area)


json_path = "/home/vargha/Desktop/export.json"
images_path = "/home/vargha/Desktop/images"
Detection = namedtuple("Detection", ["image_path", "gt", "pred"])

json_file = json.load(open(json_path))
annotations_point = json_file['annotations']
image_file = json_file['images']
examples = []

for annotation_dictionary in annotations_point:
    file_name = image_file[annotation_dictionary['image_id'] - 1]['file_name']
    bbox = annotation_dictionary['bbox']
    x, y, width, height = bbox[0], bbox[1], bbox[2], bbox[3]
    x = math.ceil(x)
    y = math.ceil(y)
    width = math.ceil(width)
    height = math.ceil(height)
    examples.append(Detection(os.path.join(images_path, file_name), [x, y, width, height], []))


for detection in examples:
    image = cv2.imread(detection.image_path)
    cv2.rectangle(image, tuple(detection.gt[:2]), tuple(detection.gt[2:]), (0, 255, 0), 2)
    cv2.rectangle(image, tuple(detection.pred[:2]), tuple(detection.pred[2:]), (0, 0, 255), 2)
    iou = intersection_over_union(detection.gt, detection.pred)
    cv2.putText(image, "IoU: {:.4f}".format(iou), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    print("{}: {:.4f}".format(detection.image_path, iou))
    cv2.imshow("Image", image)
    cv2.waitKey(0)
