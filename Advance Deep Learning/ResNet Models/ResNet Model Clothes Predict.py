# ResNet v7.ipynb

import os

import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Dense, Input, Conv2D, AveragePooling2D, add, Flatten, BatchNormalization, \
    Activation, Dropout, ActivityRegularization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.models import load_model

DATADIR = 'C:\\Users\\Vargha\\Desktop\\top'
path = '/content/data'
target = '/content/data'
CATEGORIES = ['hawaei', 'picture', 'plaid', 'plain', 'spotted', 'stripe', 'zigzag']

num_stacks = 3
num_blocks = 3
filter_sizes = [16, 32, 64]

# train = tf.keras.preprocessing.image_dataset_from_directory(DATADIR,
#                                                             label_mode="categorical",
#                                                             # class_names=CATEGORIES,
#                                                             validation_split=0.2,
#                                                             subset='training',
#                                                             batch_size=32,
#                                                             seed=42,
#                                                             image_size=(224, 224)
#                                                             )
#
# val = tf.keras.preprocessing.image_dataset_from_directory(DATADIR,
#                                                           label_mode="categorical",
#                                                           # class_names=CATEGORIES,
#                                                           validation_split=0.2,
#                                                           subset='validation',
#                                                           batch_size=32,
#                                                           seed=42,
#                                                           image_size=(224, 224)
#                                                           )
#
# def res_layer(inputs, filters, strides=1, kernel_size=3, activation='relu', batch_normalization=True):
#     x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
#                kernel_initializer='he_normal',
#                kernel_regularizer=l2(1e-3))(inputs)
#     if batch_normalization:
#         x = BatchNormalization()(x)
#     if activation is not None:
#         x = Activation(activation)(x)
#     return x
#
#
# def stack_layer(x, num_stacks, num_blocks, filters_list):
#     for stack in range(num_stacks):
#         for blocks in range(num_blocks):
#             if stack != 0 and blocks == 0:
#                 strides = 2
#             else:
#                 strides = 1
#
#             y = res_layer(x, filters_list[stack], strides=strides)
#             y = res_layer(y, filters_list[stack], strides=1, activation=None)
#
#             if stack != 0 and blocks == 0:
#                 x = res_layer(x, filters_list[stack], kernel_size=1, strides=2, activation=None,
#                               batch_normalization=False)
#             x = add([x, y])
#             x = Dropout(0.2)(x)
#             x = Activation('relu')(x)
#     return x
#
#
# inputs = Input((224, 224, 3))
# x = res_layer(inputs, filters=filter_sizes[0])
# x = stack_layer(x, num_stacks, num_blocks, filter_sizes)
# x = Dropout(0.5)(x)
# x = AveragePooling2D(8)(x)
# x = Dropout(0.5)(x)
# x = Flatten()(x)
# x = ActivityRegularization(l2=(1e-3))(x)
# outputs = Dense(7, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(1e-3))(x)
#
# model = Model(inputs, outputs)
#
# model.summary()
#
# model_checkpoint = callbacks.ModelCheckpoint('/content/drive/MyDrive/model_checkpoint(7).h5', monitor='val_accuracy',
#                                              verbose=1, save_best_only=True)
#
# tensorboard = callbacks.TensorBoard(os.path.abspath('logs'))
#
# lr_reducer = callbacks.ReduceLROnPlateau(monitor='val_accuracy',
#                                          patience=5,
#                                          factor=0.1,
#                                          verbose=1)
#
# CSVlogger = callbacks.CSVLogger('/content/drive/MyDrive/training(7).log')
#
# early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=25, verbose=1)
#
# Callbacks = [model_checkpoint, lr_reducer, CSVlogger, tensorboard, early_stopping]


# model.fit(train, batch_size=32, epochs=400, validation_data=val, verbose=1, callbacks=Callbacks)

new_model = load_model(os.path.abspath('ResNet on Clothes.h5'))
new_model.compile(loss="categorical_crossentropy", optimizer=Adam(0.001), metrics=["accuracy"])

img_path = ('C:\\Users\\Vargha\\Desktop\\jack-and-jones_244124_12181602_WHISPER-WHITE_20201215T092917_01.jpg')
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
    i += 1

prediction = new_model.predict(img_preprocessed)
print(prediction)



# print(decode_predictions(prediction, top=3)[0])
