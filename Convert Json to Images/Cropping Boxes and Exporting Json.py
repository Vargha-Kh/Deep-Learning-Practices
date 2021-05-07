import os
import math
import cv2
import json


def splitting_images(json_path, images_dir, json_export, output_images):
    json_file = json.load(open(json_path))
    annotations_point = json_file['annotations']
    image_file = json_file['images']
    number = 0
    images = []
    annotations = []
    for annotation_dictionary in annotations_point:
        file_name = image_file[annotation_dictionary['image_id'] - 1]['file_name']
        bbox = annotation_dictionary['bbox']
        segments = annotation_dictionary['segmentation']
        x, y, width, height = bbox[0] - 5, bbox[1] - 5, bbox[2] + 10, bbox[3] + 10
        if x < 0 or y < 0:
            x = x + 5
            y = y + 5
        bbox_new = [bbox[0] - x, bbox[1] - y, bbox[2], bbox[3]]
        area = bbox[2] * bbox[3]
        for segment in segments:
            for i in range(0, len(segment), 2):
                segment[i] -= x
                segment[i + 1] -= y
        x = math.ceil(x)
        y = math.ceil(y)
        width = math.ceil(width)
        height = math.ceil(height)
        img = os.path.join(images_dir, file_name)
        img = cv2.imread(img)
        crop_img = img[y:y + height, x:x + width]
        file_name = str(number) + '_' + file_name
        cv2.imwrite(os.path.join(output_images, file_name), crop_img)

        annotations.append(
            {'id': number, 'iscrowd': annotation_dictionary['iscrowd'], 'image_id': number + 1,
             'category_id': annotation_dictionary['category_id'],
             'segmentation': segments, 'bbox': bbox_new, 'area': area})
        number += 1
        images.append({'id': number, 'width': width, 'height': height, 'file_name': file_name})
        new_annotations = {'info': json_file['info'], 'images': images, 'annotations': annotations,
                           'categories': json_file['categories']}
        with open(json_export + 'export.json', 'w') as outfile:
            json.dump(new_annotations, outfile)


json_file_path = "/home/vargha/Desktop/DORNA.1.FCSEG.json"
output_path = "/home/vargha/Desktop/Export"
images_path = "/home/vargha/Desktop/images"
json_export_dir = "/home/vargha/Desktop/Export/"
splitting_images(json_file_path, images_path, json_export_dir, output_path)
