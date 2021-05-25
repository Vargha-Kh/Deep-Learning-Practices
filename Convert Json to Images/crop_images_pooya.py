import os
import math
import cv2
import numpy as np
import json
from multiprocessing import Pool

max_x = 20
max_y = 20


def main_func(in_args):
    annotation_dictionary, number, img_dict, images_dir, output_images = in_args
    x_extend = int(np.random.randint(0, max_x, 1))
    y_extend = int(np.random.randint(0, max_y, 1))
    file_name = img_dict[annotation_dictionary['image_id']]['file_name']
    bbox = annotation_dictionary['bbox']
    segments = annotation_dictionary['segmentation']
    x, y, width, height = bbox[0] - x_extend, bbox[1] - y_extend, bbox[2] + 2 * x_extend, bbox[3] + 2 * y_extend
    x = 0 if x < 0 else x
    y = 0 if y < 0 else y
    bbox_new = [bbox[0] - x, bbox[1] - y, bbox[2], bbox[3]]
    area = bbox[2] * bbox[3]
    new_segments = []
    for segment in segments:
        a = np.array(segment)
        a = a - np.array([x, y] * int(a.size // 2))
        new_segments.append(a.tolist())
    x, y, width, height = int(x), int(y), int(width), int(height)
    img_path = os.path.join(images_dir, file_name)
    img = cv2.imread(img_path)
    crop_img = img[y:y + height, x:x + width]
    img_h, img_w, _ = crop_img.shape
    width = img_w if width > img_w else width
    height = img_h if height > img_h else height
    file_name = str(number) + '_' + file_name
    file_path = os.path.join(output_images, file_name)
    try:
        cv2.imwrite(file_path, crop_img)
    except:
        print('wrong')
        return None
    return area, bbox_new, file_name, height, new_segments, width, number, annotation_dictionary


def splitting_images(json_path, images_dir, json_export, output_images):
    os.system(f'rm -rf {output_images}')
    os.makedirs(output_images)
    with open(json_path) as f:
        json_file = json.load(f)
    annotations_point = json_file['annotations']
    image_file = json_file['images']
    img_dict = {img_file['id']: img_file for img_file in image_file}
    images = []
    annotations = []
    pool = Pool(processes=os.cpu_count() - 1)
    len_annotations = len(annotations_point)
    pool_out = pool.imap(main_func, zip(annotations_point,
                                        range(1, len_annotations + 1),
                                        [img_dict] * len_annotations,
                                        [images_dir] * len_annotations,
                                        [output_images] * len_annotations))
    c = 0
    for out in pool_out:
        if out is None:
            continue
        print(f"{c}/{len(annotations_point)} is done!")
        area, bbox_new, file_name, height, new_segments, width, number, annotation_dictionary = out
        annotations.append(
            {'id': number, 'iscrowd': annotation_dictionary['iscrowd'], 'image_id': number,
             'category_id': annotation_dictionary['category_id'],
             'segmentation': new_segments, 'bbox': bbox_new, 'area': area})
        images.append({'id': number, 'width': width, 'height': height, 'file_name': file_name})
        if c % 1000 == 0:
            new_annotations = {'info': json_file['info'], 'images': images, 'annotations': annotations,
                               'categories': json_file['categories']}
            with open(json_export, 'w') as outfile:
                json.dump(new_annotations, outfile)
        c += 1
    new_annotations = {'info': json_file['info'], 'images': images, 'annotations': annotations,
                       'categories': json_file['categories']}
    with open(json_export, 'w') as outfile:
        json.dump(new_annotations, outfile)


splitting_images("/home/symo-pouya/projects/demo-v1/segment/cloth.json",
                 "/home/symo-pouya/projects/demo-v1/segment/cloth",
                 "/home/symo-pouya/projects/demo-v1/segment/cloth_crop.json",
                 "/home/symo-pouya/projects/demo-v1/segment/cloth_crop")
splitting_images("val_cloth.json", "val_cloth", "val_cloth_crop.json", "val_cloth_crop")
