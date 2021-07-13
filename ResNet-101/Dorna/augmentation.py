import imgaug.augmenters as iaa
import numpy as np
from torchvision import transforms, datasets

seq_1 = iaa.Sequential([
    iaa.Fliplr(1.0),
    iaa.Rotate(rotate=(-30, 30)),
    iaa.ScaleX(scale=(0.8, 1.0)),
    iaa.MultiplyHue((0.5, 1.5)),
])

seq_2 = iaa.Sequential([
    iaa.PiecewiseAffine(scale=0.05),
    iaa.Rotate((-30, 30)),
    iaa.TranslateX(px=(-30, 30)),
    iaa.TranslateY(px=(-30, 30)),
    iaa.Grayscale(0.6)
])

seq_3 = iaa.Sequential([
    iaa.Crop(percent=0.1),
    iaa.ScaleX(scale=(0.8, 0.9)),
    iaa.Fliplr(0.5),
    iaa.ChangeColorTemperature((1100, 10000)),
    iaa.MultiplyHue((0.5, 1.5))
])

seq_4 = iaa.Sequential([
    iaa.Affine((1.1, 1.3)),
    iaa.Rotate((-5, 5)),
    iaa.TranslateX(percent=0.04),
    iaa.ShearX((-15, 15))
])

seq_5 = iaa.Sequential([
    iaa.Fliplr(1.0),
    iaa.Crop(percent=0.02),
    iaa.TranslateX(percent=[-0.1, 0.1]),
    iaa.ChangeColorTemperature((6000, 11000))
])

seq_6 = iaa.Sequential([
    iaa.PiecewiseAffine(scale=0.05),
    iaa.Rotate((-30, 30)),
    iaa.TranslateX(px=(-30, 30)),
    iaa.TranslateY(px=(-30, 30)),
])
seq = {1: seq_1, 2: seq_2, 3: seq_3, 4: seq_4, 5: seq_5, 6: seq_6}
SEQ = [seq_1, seq_2, seq_3, seq_4, seq_5, seq_6]
aug_dict = {
    'hawaei': [seq_1, seq_2, seq_3, seq_4, seq_5],
    'picture': [seq_1, seq_2, seq_3, seq_4, seq_5],
    'plaid': [seq_1, seq_3, seq_4, seq_5],
    'plain': [seq_1, seq_2, seq_3, seq_4, seq_5],
    'spotted': [seq_1, seq_3, seq_4, seq_5, seq_6],
    'stripe': [seq_1, seq_3, seq_4, seq_5],
    'zigzag': [seq_1, seq_2, seq_3, seq_4, seq_5],

}
classes = {'hawaei', 'picture', 'plaid', 'plain', 'spotted', 'stripe', 'zigzag'}
data_dir = '/home/dorna/symo/Dorna/data'


def Aug(data_dir):
    for s in SEQ:
        tf = transforms.Compose([s.augment_image, transforms.ToTensor()])
        train_transforms = transforms.Compose([transforms.Resize((224, 224))])
        augmentation = []
        images = datasets.ImageFolder(data_dir, transform=train_transforms)
        imgs = [list(i) for i in images]
        a = [None] * len(images.classes)
        for n in range(len(images.classes)):
            if n == 0:
                a[n] = images.targets.count(n)
            else:
                a[n] = images.targets.count(n) + a[n - 1]
        a.insert(0, 0)
        for img in imgs:
            # convertiong pillow to np array
            img[0] = np.array(img[0])
        for count in range(len(a)-1):
            if s in aug_dict[images.classes[count]]:
                for im in imgs[a[count]:a[count + 1]]:
                        tf(im[0])
        augmentation.append(imgs)
    return augmentation