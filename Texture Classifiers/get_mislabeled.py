import os
import shutil
from resnet101.model import get_model
from augmentation import DataGenerator
from tensorflow.keras import optimizers
import numpy as np

# parameters
train = True
res_path = r'resnet101/resnet101.h5'

num_classes = 7

lr = 1e-3
size = (224, 224)
batch_size = 32
epochs = 300
augment = True

class_names = {'plaid': 0,
               'spotted': 1,
               'hawaei': 2,
               'stripe': 3,
               'plain': 4,
               'picture': 5,
               'zigzag': 6}
class_names = {v: k for k, v in class_names.items()}
model = get_model(num_classes=num_classes, weights=res_path)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizers.Adam(lr=lr),
              metrics=['accuracy'])

print(model.summary())

train_generator = DataGenerator('top', r'.', batch_size=batch_size, dim=size, augment=augment, return_path=True)
val_generator = DataGenerator('validation/top', r'.', batch_size=batch_size, dim=size, augment=False, val=True,
                              return_path=True)


def get_wrong(generator, wrong_path):
    print('started generation')
    os.system(f'rm -rf {wrong_path}')
    os.makedirs(wrong_path)
    c = 1
    for x, true_y, paths in generator:
        pred_y = model.predict(x)
        pred_y = np.argmax(pred_y, axis=1)
        for e, (y1, y2) in enumerate(zip(pred_y, true_y)):
            y1 = int(y1)
            y2 = int(y2)
            if y1 != y2:
                name = os.path.split(paths[e])[-1]
                name = name[:-4] + ' has been labeled ' + class_names[y1] + ' while it is ' + class_names[y2] + '.jpg'
                shutil.copy(paths[e], os.path.join(wrong_path, name))
                print(c, ' ', name)
                c += 1
    print('finished generation')


get_wrong(val_generator, wrong_path='resnet101/wrong_val')
get_wrong(train_generator, wrong_path='resnet101/wrong_train')
