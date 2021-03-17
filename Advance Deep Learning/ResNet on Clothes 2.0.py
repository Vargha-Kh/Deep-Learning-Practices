import os
import pickle
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.tests.test_base import K
from tensorflow.keras.layers import Dense, Input, Conv2D, AveragePooling2D, add, Flatten, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.tests.test_base import K
from keras.models import load_model

DATADIR = '/content/data'
CATEGORIES = ['hawaei', 'picture', 'plaid', 'plain', 'spotted', 'stripe', 'zigzag']

train = tf.keras.preprocessing.image_dataset_from_directory(DATADIR,
                                                            label_mode="categorical",
                                                            # class_names=CATEGORIES,
                                                            validation_split=0.2,
                                                            subset='training',
                                                            batch_size=32,
                                                            seed=42,
                                                            image_size=(224, 224)
                                                            )

val = tf.keras.preprocessing.image_dataset_from_directory(DATADIR,
                                                          label_mode="categorical",
                                                          # class_names=CATEGORIES,
                                                          validation_split=0.2,
                                                          subset='validation',
                                                          batch_size=32,
                                                          seed=42,
                                                          image_size=(224, 224)
                                                          )

num_stacks = 3
num_blocks = 3
filter_sizes = [16, 32, 64]


def res_layer(inputs, filters, strides=1, kernel_size=3, activation='relu', batch_normalization=True):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4))(inputs)
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


inputs = Input((224, 224, 3))
x = res_layer(inputs, filters=filter_sizes[0])
x = stack_layer(x, num_stacks, num_blocks, filter_sizes)
x = AveragePooling2D(8)(x)
x = Flatten()(x)
outputs = Dense(7, activation='softmax', kernel_initializer='he_normal')(x)
model = Model(inputs, outputs)
# model = load_model("/content/drive/MyDrive/model_checkpoint.h5")

model.summary()


def lr_schedule(epoch):
    lr = 0.01
    if epoch > 20:
        lr *= 1e-2
    return lr


model_checkpoint = callbacks.ModelCheckpoint('/content/drive/MyDrive/model_checkpoint.h5', monitor='val_accuracy', verbose=1, save_best_only=True)

tensorboard = callbacks.TensorBoard(os.path.abspath('logs'))

lr_scheduler = callbacks.LearningRateScheduler(lr_schedule, verbose=1)

lr_reducer = callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                         patience=5,
                                         factor=0.1,
                                         verbose=1)

CSVlogger = callbacks.CSVLogger('/content/drive/MyDrive/training.log')

early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=1)

# class LRTensorBoard(callbacks.TensorBoard):
#     def __init__(self, log_dir, **kwargs):
#         super().__init__(log_dir=log_dir, **kwargs)
#
#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#         logs.update({'lr': eval(self.model.optimizer.lr)})
#         super().on_epoch_end(epoch, logs)
#
#
Callbacks = [model_checkpoint, lr_reducer, CSVlogger, tensorboard]

model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])

history = model.fit(train, batch_size=32, epochs=40, validation_data=val, verbose=1, callbacks=Callbacks)
