from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from imutils import paths
from tensorflow.keras.layers import Conv2D, Input, Flatten, MaxPooling2D, Dense, BatchNormalization, Dropout, Activation, add, AveragePooling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import argparse
import imutils
import cv2
import os

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def resnet(shape=(32,32,1)):
    num_stacks = 1
    num_blocks = 1
    filter_sizes = [16, 32, 64]

    def res_layer(inputs, filters, strides=1, kernel_size=3, activation='relu', batch_normalization=True):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                   kernel_initializer='he_normal')(inputs)
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

    inputs = Input(shape=shape)
    x = res_layer(inputs, filters=filter_sizes[0])
    x = BatchNormalization()(x)
    x = stack_layer(x, num_stacks, num_blocks, filter_sizes)
    x = BatchNormalization()(x)
    x = AveragePooling2D(8)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
    return Model(inputs, outputs)

def leNet(shape=(32, 32, 1)):
    inputs = Input(shape=shape)
    x = Conv2D(filters=20, kernel_size=5, padding='same', activation='relu')(inputs)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    x = Conv2D(filters=50, kernel_size=5, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    x = Flatten()(x)
    x = Dense(120, activation='relu')(x)
    x = Dense(84, activation='relu')(x)
    outputs = Dense(2, activation='softmax')(x)
    return Model(inputs, outputs)


# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True,
#                 help="path to input dataset of faces")
# ap.add_argument("-m", "--model", required=True,
#                 help="path to output model")
# args = vars(ap.parse_args())
data = []
labels = []

dataset_dir = "/home/vargha/Desktop/datasets/datasets"
output = "/home/vargha/Desktop/model.hdf5"

for imagePath in sorted(list(paths.list_images(dataset_dir))):
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=32)
    image = img_to_array(image)
    data.append(image)
    label = imagePath.split(os.path.sep)[-3]
    label = "smiling" if label == "positives" else "not_smiling"
    labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
le = LabelEncoder().fit(labels)
labels = to_categorical(le.transform(labels), 2)
classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=0.20, stratify=labels, random_state=42)

print("[INFO] compiling model...")
model = resnet()
model.compile(loss="binary_crossentropy", optimizer="adam",
              metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=64, epochs=15, verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=le.classes_))
# save the model to disk
print("[INFO] serializing network...")
model.save(output)

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 15), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 15), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 15), H.history["accuracy"], label="acc")
plt.plot(np.arange(0, 15), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
