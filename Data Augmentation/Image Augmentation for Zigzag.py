import os
import cv2
import imgaug.augmenters as iaa

path = 'C:\\Users\\Vargha\\Desktop\\zigzag clothes'
target = 'C:\\Users\\Vargha\\Desktop\\temp'

# for i in range(1, 6):
for image in (os.listdir(path)):
    img = cv2.imread(os.path.join(path, image))
    h = img.shape[0]
    w = img.shape[1]
    c = img.shape[2]
    i = 3
    if i == 1:
        seq = iaa.Sequential([
            # For Zigzag Clothes
            iaa.Rotate((-5, 5)),
            iaa.Fliplr(1.0),
            iaa.ScaleX(scale=(1.3, 1.5)),
            iaa.PiecewiseAffine(scale=(0.05)),
            iaa.MultiplyHue((0.5, 1.5)),
            iaa.ChangeColorTemperature((6000, 11000))
        ])
        images_aug = seq(image=img)
        image_split = image.split(sep='.')
        image_name = image_split[0] + str(i) + '.' + 'png'
        cv2.imwrite(os.path.join(target, image_name), images_aug)
    if i == 2:
        seq = iaa.Sequential([
            # For Zigzag Pattern
            iaa.Fliplr(1.0),
            iaa.ScaleX(scale=(1.3, 1.5)),
            iaa.MultiplyHue((0.5, 1.5)),
            iaa.ChangeColorTemperature((6000, 11000))
        ])
        images_aug = seq(image=img)
        image_split = image.split(sep='.')
        image_name = image_split[0] + str(i) + '.' + 'png'
        cv2.imwrite(os.path.join(target, image_name), images_aug)

    if i == 3:
        seq = iaa.Sequential([
            # For Zigzag Clothes(2)
            iaa.PadToFixedSize(width=w + 100, height=100 + h, position='center'),
            iaa.Rotate((-5, 5)),
            iaa.Fliplr(1.0),
            # iaa.TranslateX(percent=[-0.1, 0.1]),
            iaa.ScaleX(scale=(1.3, 1.5)),
            iaa.MultiplyHue((0.5, 1.5)),
            iaa.ChangeColorTemperature((6000, 11000))
        ])
        images_aug = seq(image=img)
        image_split = image.split(sep='.')
        image_name = image_split[0] + str(i) + '.' + 'png'
        cv2.imwrite(os.path.join(target, image_name), images_aug)
