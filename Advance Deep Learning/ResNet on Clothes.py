import os
import pickle
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Input, Conv2D, AveragePooling2D, add, Flatten, BatchNormalization, Activation
from keras.models import Model
from keras.regularizers import L2
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

DATADIR = 'C:\\Users\\Vargha\\Desktop\\top'
CATEGORIES = ['hawaei', 'logo', 'picture', 'plaid', 'plain', 'spotted', 'stripe', 'zigzag']

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img))
        # plt.imshow(img_array)
        # plt.show()
        break
    break

IMG_SIZE = 32
training_data = []


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])


create_training_data()

random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array(y)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

input_shape = X.shape[1:]

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

x_train, y_train, x_test, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# mean_pixel = np.mean(x_train, axis=0)
# x_train = x_train - mean_pixel
# x_test = x_test - mean_pixel

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_stacks = 3
num_blocks = 3
filter_sizes = [16, 32, 64]


def res_layer(inputs, filters, strides=1, kernel_size=3, activation='relu', batch_normalization=True):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=L2(1e-4))(inputs)
    if batch_normalization:
        x = BatchNormalization()(x)
    if activation is not None:
        x = Activation(activation)(x)
    return x


def stack_layer(x, num_stacks, num_blocks, filters_list):
    for stack in range(num_stacks):
        for blocks in range(num_blocks):
            if stack != 0 and blocks == 0:
                strides = 2
            else:
                strides = 1

            y = res_layer(x, filters_list[stack], strides=strides)
            y = res_layer(y, filters_list[stack], strides=1, activation=None)

            if stack != 0 and blocks == 0:
                x = res_layer(x, filters_list[stack], kernel_size=1, strides=2, activation=None,
                              batch_normalization=False)
            x = add([x, y])
            x = Activation('relu')(x)
    return x


inputs = Input((32, 32, 3))
x = res_layer(inputs, filters=filter_sizes[0])
x = stack_layer(x, num_stacks, num_blocks, filter_sizes)
x = AveragePooling2D(8)(x)
x = Flatten()(x)
outputs = Dense(8, activation='softmax', kernel_initializer='he_normal')(x)
model = Model(inputs, outputs)
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=32, epochs=40, validation_data=(x_test, y_test), verbose=1)
