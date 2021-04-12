import Augmentor
import numpy
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
import os
import imgaug as ia
import imgaug.augmenters as iaa
import tensorflow as tf
import cv2
import shutil
import imagehash
from PIL.Image import core as _imaging

os.chdir('C:\\Users\\Vargha\\Desktop\\images')
path = 'C:\\Users\\Vargha\\Desktop\\stripe png'
target = 'C:\\Users\\Vargha\\Desktop\\Augmented Stripe'
for i in range(1, 6):
    for image in (os.listdir(path)):
        img = cv2.imread(os.path.join(path, image))
        h = img.shape[0]
        w = img.shape[1]
        c = img.shape[2]
        if i == 1:
            seq = iaa.Sequential([
                # 1
                iaa.PadToFixedSize(width=w + 100, height=h + 100),
                iaa.Fliplr(1.0),
                iaa.Rotate(rotate=(-30, 30)),
                iaa.ScaleX(scale=(0.8, 1.0)),
                iaa.MultiplyHue((0.5, 1.5)),
            ])
            images_aug = seq(image=img)
            image_split = image.split(sep='.')
            image_name = image_split[0] + str(i) + '.' + image_split[1]
            cv2.imwrite(os.path.join(target, image_name), images_aug)
        if i == 2:
            seq = iaa.Sequential([
                # 2
                iaa.PiecewiseAffine(scale=(0.05)),
                iaa.PadToFixedSize(width=w + 100, height=100 + h),
                iaa.Rotate((-30, 30)),
                iaa.TranslateX(px=(-30, 30)),
                iaa.TranslateY(px=(-30, 30)),
                iaa.Grayscale(0.5)
            ])
            images_aug = seq(image=img)
            image_split = image.split(sep='.')
            image_name = image_split[0] + str(i) + '.' + image_split[1]
            cv2.imwrite(os.path.join(target, image_name), images_aug)
        if i == 3:
            seq = iaa.Sequential([
                # 3
                iaa.Crop(percent=0.1),
                iaa.ScaleX(scale=(0.8, 0.9)),
                iaa.Fliplr(1.0),
                iaa.ChangeColorTemperature((1100, 10000)),
                iaa.MultiplyHue((0.5, 1.5))
            ])
            images_aug = seq(image=img)
            image_split = image.split(sep='.')
            image_name = image_split[0] + str(i) + '.' + image_split[1]
            cv2.imwrite(os.path.join(target, image_name), images_aug)
        if i == 4:
            seq = iaa.Sequential([
                # 4
                iaa.PadToFixedSize(width=w + 200, height=200 + h),
                iaa.Affine((1.1, 1.3)),
                iaa.Rotate((-5, 5)),
                iaa.TranslateX(percent=0.04),
                iaa.ShearX((-15, 15))
            ])
            images_aug = seq(image=img)
            image_split = image.split(sep='.')
            image_name = image_split[0] + str(i) + '.' + image_split[1]
            cv2.imwrite(os.path.join(target, image_name), images_aug)

        if i == 5:
            seq = iaa.Sequential([
                # 5
                iaa.PadToFixedSize(width=w + 100, height=h + 100),
                iaa.Fliplr(1.0),
                iaa.Crop(percent=0.02),
                iaa.TranslateX(percent=[-0.1, 0.1]),
                iaa.ChangeColorTemperature((6000, 11000))
            ])
            images_aug = seq(image=img)
            image_split = image.split(sep='.')
            image_name = image_split[0] + str(i) + '.' + image_split[1]
            cv2.imwrite(os.path.join(target, image_name), images_aug)

        # p = Augmentor.Pipeline('C:\\Users\\Vargha\\Desktop\\images', output_directory='C:\\Users\\Vargha\\Desktop\\New folder')
        # p.crop_centre(1, percentage_area=0.75)
        # p.scale(1, 0.5)
        # p.flip_left_right(1)
        # p.rotate(1, -30, 30)
        # p.sample(10)

        # datagen = ImageDataGenerator(
        #     rotation_range=30,
        #     # width_shift_range=[0.1, 0.2],
        #     # height_shift_range=[0.1, 0.2],
        #     # rescale=1. / 255,
        #     # shear_range=[0.2, 0.3],
        #     # zoom_range=[0.2, 0.5],
        #     horizontal_flip=True,
        #     # fill_mode='nearest'
        # )
        #
        # for filename in (os.listdir()):
        #     img = Image.open(filename)
        #     x = img_to_array(img)
        #     x = x.reshape((1,) + x.shape)
        #
        #     i = 0
        #     for batch in datagen.flow(x, batch_size=1,
        #                               save_to_dir='C:\\Users\\Vargha\\Desktop\\New folder', save_prefix='AUG',
        #                               save_format='jpeg'):
        #         i += 1
        #         if i > 20:
        #             break

        # def random_crop(image):
        #     height, width = image.shape[:2]
        #     random_array = numpy.random.random(size=4);
        #     w = int((width*0.5) * (1+random_array[0]*0.5))
        #     h = int((height*0.5) * (1+random_array[1]*0.5))
        #     x = int(random_array[2] * (width-w))
        #     y = int(random_array[3] * (height-h))
        #
        #     image_crop = image[y:h+y, x:w+x, 0:3]
        #     image_crop = numpy.resize(image_crop, image.shape)
        #     return image_crop
