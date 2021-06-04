import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Activation, Flatten, DepthwiseConv2D, BatchNormalization, \
    AveragePooling2D, Dense
from tensorflow.keras.datasets import mnist
import numpy as np
from keras.utils import to_categorical
from mobilenet import MobileNet

# Preprocessing Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

x_train.astype(np.float32) / 255
x_test.astype(np.float32) / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


def separable_conv(x, filters, strides, alpha=0.1):
    x = DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = BatchNormalization(momentum=0.9997)(x)
    x = Activation('relu')(x)
    x = Conv2D(np.floor(filters * alpha), strides=strides, kernel_size=(1, 1))(x)
    x = BatchNormalization(momentum=0.9997)(x)
    x = Activation('relu')(x)
    return x


def conv(x, filters, kernel_size=1, strides=1, alpha=0.1):
    x = Conv2D(np.floor(filters * alpha), kernel_size=kernel_size, use_bias=False, strides=strides, padding='same')(x)
    x = BatchNormalization(momentum=0.9997)(x)
    x = Activation('relu')(x)
    return x


# Model layers
input_layer = Input((28, 28, 1))
x = conv(input_layer, 32, 3, 2)
x = separable_conv(x, 32, 1)
x = conv(x, 64, 1)
x = separable_conv(x, 64, 2)
x = conv(x, 128, 1)
x = separable_conv(x, 128, 1)
x = conv(x, 128, kernel_size=1)
x = separable_conv(x, 128, strides=2)
x = conv(x, 256, kernel_size=1)
x = separable_conv(x, 256, strides=1)
x = conv(x, 256, kernel_size=1)
x = separable_conv(x, 256, strides=2)
x = conv(x, 512, kernel_size=1)
for i in range(5):
    x = separable_conv(x, 512, strides=1)
    x = conv(x, 512, kernel_size=1)
x = separable_conv(x, 512, strides=2)
x = conv(x, 1024, kernel_size=1)
x = AveragePooling2D(pool_size=(7, 7))(x)
x = Flatten()(x)
outputs = Dense(10, activation='softmax')(x)
model = tf.keras.models.Model(input_layer, outputs)

# Training model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=100, validation_split=0.2, validation_data=(x_test, y_test),
          verbose=1)

# Evaluating Model
model.evaluate(x_test, y_test)
