import json
import cv2
import shutil
import os


def splitting_categories(json_path, json_export, images_dir, output_images):
    tshirt_annotations = []
    shirt_annotations = []
    pants_annotations = []
    tshirt_images = []
    shirt_images = []
    pants_images = []
    json_file = json.load(open(json_path))
    for annotation in json_file["annotations"]:
        if annotation['category_id'] == 1:
            shirt_annotations.append(annotation)
            shirt_images.append(json_file["images"][annotation['image_id'] - 1])
        elif annotation['category_id'] == 2:
            tshirt_annotations.append(annotation)
            tshirt_images.append(json_file["images"][annotation['image_id'] - 1])
        elif annotation['category_id'] == 3:
            pants_annotations.append(annotation)
            pants_images.append(json_file["images"][annotation['image_id'] - 1])

    annotations_shirt = {'info': json_file['info'], 'images': shirt_images, 'annotations': shirt_annotations,
                         'categories': json_file['categories']}

    annotations_tshirt = {'info': json_file['info'], 'images': tshirt_images, 'annotations': tshirt_annotations,
                          'categories': json_file['categories']}

    annotations_pants = {'info': json_file['info'], 'images': pants_images, 'annotations': pants_annotations,
                         'categories': json_file['categories']}

    for shirt_image in shirt_images:
        file_name = shirt_image['file_name']
        shutil.copyfile(os.path.join(images_dir, file_name), os.path.join(output_images['shirt'], file_name))

    for tshirt_image in tshirt_images:
        file_name = tshirt_image['file_name']
        shutil.copyfile(os.path.join(images_dir, file_name), os.path.join(output_images['tshirt'], file_name))

    for pant_image in pants_images:
        file_name = pant_image['file_name']
        shutil.copyfile(os.path.join(images_dir, file_name), os.path.join(output_images['pants'], file_name))

    with open(json_export + 'export_shirt.json', 'w') as outfile:
        json.dump(annotations_shirt, outfile)

    with open(json_export + 'export_tshirt.json', 'w') as outfile:
        json.dump(annotations_tshirt, outfile)

    with open(json_export + 'export_pants.json', 'w') as outfile:
        json.dump(annotations_pants, outfile)


json_file_path = "/home/vargha/Desktop/VARGHA.55.FCSEG.json"

output_path = {
    "shirt": "/home/vargha/Desktop/Export_images/shirts/",
    "tshirt": "/home/vargha/Desktop/Export_images/tshirts/",
    "pants": "/home/vargha/Desktop/Export_images/pants/"
}
images_path = "/home/vargha/Desktop/images"
json_export_dir = "/home/vargha/Desktop/Export/"
splitting_categories(json_file_path, json_export_dir, images_path, output_path)
