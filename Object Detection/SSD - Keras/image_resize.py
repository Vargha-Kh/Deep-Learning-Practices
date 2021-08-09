from PIL import Image
import os
import cv2
import PIL
import glob

os.chdir('/home/vargha/Desktop/images')
saved_dir = "/home/vargha/Desktop/chapter11-detection/images/"
for image_name in (os.listdir()):
    image = cv2.imread(image_name)
    resized_image = cv2.resize(image, (400, 300))
    cv2.imwrite(os.path.join(saved_dir, image_name), resized_image)
