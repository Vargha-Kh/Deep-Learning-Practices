import tensorflow as tf
import pandas as pd
import numpy as np
import os
import keras
import tensorflow.keras as K
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Input, BatchNormalization, Dropout
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import callbacks

DATADIR = '/content/data'
path = '/content/data'
target = '/content/data'
CATEGORIES = ['hawaei', 'picture', 'plaid', 'plain', 'spotted', 'stripe', 'zigzag']


train = tf.keras.preprocessing.image_dataset_from_directory(DATADIR,
                                                            label_mode="categorical",
                                                            # class_names=CATEGORIES,
                                                            validation_split=0.2,
                                                            subset='training',
                                                            batch_size=128,
                                                            seed=42,
                                                            image_size=(224, 224)
                                                            )

val = tf.keras.preprocessing.image_dataset_from_directory(DATADIR,
                                                          label_mode="categorical",
                                                          # class_names=CATEGORIES,
                                                          validation_split=0.2,
                                                          subset='validation',
                                                          batch_size=128,
                                                          seed=42,
                                                          image_size=(224, 224)
                                                          )


inputs = Input((224, 224, 3))
base_model = MobileNet(weights='imagenet', include_top=False, input_tensor=inputs)
for layer in base_model.layers:
    layer.trainable = False
for i, layer in enumerate(base_model.layers):
        print(i, layer.name, "-", layer.trainable)
x = Flatten()(base_model.output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
outputs = Dense(7, activation='softmax', kernel_initializer='he_normal',  kernel_regularizer=l2(1e-2))(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

model_checkpoint = callbacks.ModelCheckpoint('/content/drive/MyDrive/transfer_learning_MobileNet.h5', monitor='val_accuracy',
                                             verbose=1, save_best_only=True)

tensorboard = callbacks.TensorBoard('/content/drive/MyDrive/transfer_learning_logs_MobileNet')

lr_reducer = callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                         patience=5,
                                         factor=0.1,
                                         verbose=1)

CSVlogger = callbacks.CSVLogger('/content/drive/MyDrive/training_MobileNet.log')

early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=35, verbose=1)

Callbacks = [model_checkpoint, lr_reducer, CSVlogger, tensorboard, early_stopping]

history = model.fit(train, batch_size=128, epochs=400, validation_data=val, verbose=1, callbacks=Callbacks)
