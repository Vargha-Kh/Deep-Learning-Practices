import numpy as np
from keras.datasets import mnist
from keras.models import Model
from keras.utils import to_categorical
from keras.layers import Conv2D, AveragePooling2D, Dense, Flatten, Input

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(y_train)

image_size = x_train.shape[1]
x_train = x_train.reshape((-1, image_size, image_size, 1))
x_test = x_test.reshape((-1, image_size, image_size, 1))
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

inputs = Input(shape=(32, 32, 1))
x = inputs
x = Conv2D(filters=6, kernel_size=5)(x)
x = AveragePooling2D(pool_size=2, strides=2)(x)
x = Conv2D(filters=16, kernel_size=5)(x)
x = AveragePooling2D(pool_size=2, strides=2)(x)
x = Flatten()(x)
x = Dense(120, activation='relu')(x)
x = Dense(84, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)


model = Model(inputs, outputs, name="LeNet-5")
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test), verbose=1)
