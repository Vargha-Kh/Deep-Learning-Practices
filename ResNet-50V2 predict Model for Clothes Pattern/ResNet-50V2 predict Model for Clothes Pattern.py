from keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.models import Model, load_model
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import save_img
import os

# Path folder should have all the classes as subfolder
path = 'C:\\Users\\Vargha\\Desktop\\New folder'
target = 'C:\\Users\\Vargha\\Desktop\\target'
unknown = 'C:\\Users\\Vargha\\Desktop\\unknown'
CATEGORIES = ['hawaei', 'picture', 'plaid', 'plain', 'spotted', 'stripe', 'zigzag']
new_model = load_model('C:\\Users\\Vargha\\Desktop\\transfer_learning.h5')

for new_image in (os.listdir(path)):
    img_path = (os.path.join(path, new_image))
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    classes = np.argmax(new_model.predict(img_preprocessed))
    classes = classes.astype(np.int)
    i = 0
    for name in CATEGORIES:
        if classes == i:
            print('Class:', name)
            image_split = new_image.split(sep='.')
            image_name = image_split[0] + str(i) + '.' + image_split[1]
            # prediction = new_model.predict(img_preprocessed)
            predict = np.max(new_model.predict(img_preprocessed))
            if predict > 0.25:
                save_img(path=os.path.join(target, name, '{image_name}.jpg'.format(image_name=image_name)), x=img)
            else:
                save_img(os.path.join(unknown, '{image_name}.jpg'.format(image_name=image_name)), img)
        i += 1

