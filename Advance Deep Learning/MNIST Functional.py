import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Input, Dropout

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))

x_train = x_train.astype(np.float32) / 255
x_test = x_test.astype(np.float32) / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

inputs = Input(shape=(784,), name="input")
x = inputs
x = Dense(units=256, activation='relu', name="Layer_1")(x)
x = Dropout(0.5)(x)
x = Dense(units=256, activation='relu', name="Layer_2")(x)
x = Dropout(0.5)(x)
outputs = Dense(units=10, activation='softmax', name="outputs")(x)
model = Model(inputs=inputs, outputs=outputs, name="mnist_model")
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=x_train, y=y_train, epochs=20, validation_data=(x_test, y_test), batch_size=32, verbose=1)
