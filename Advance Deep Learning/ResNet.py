from keras.utils import to_categorical, plot_model
from keras.layers import Dense, AveragePooling2D, Flatten, add, Input, Conv2D, BatchNormalization, Activation
from keras.models import Model
from keras.regularizers import L2
from keras.datasets import cifar10
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

mean = np.mean(x_train, axis=0)
x_train -= mean
x_test -= mean

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


filter_sizes = [16, 32, 64]
num_stack = 3
num = 3


def resnet_layer(filter_size, input_layer, kernel_size=3, strides=1, activation='relu', batch_normalization=True):
    x = Conv2D(filters=filter_size, kernel_size=kernel_size, strides=strides, padding='same',
               kernel_regularizer=L2(1e-4), kernel_initializer='he_normal')(input_layer)
    if batch_normalization:
        x = BatchNormalization()(x)
    if activation is not None:
        x = Activation('relu')(x)

    return x


def residual_stack(x, num_stack, num, filter_sizes):
    for stack in range(num_stack):
        for block in range(num):
            if stack != 0 and block == 0:
                strides = 2
            else:
                strides = 1

            # Residual Layer
            y = resnet_layer(input_layer=x, filter_size=filter_sizes[stack], strides=strides)
            y = resnet_layer(input_layer=y, filter_size=filter_sizes[stack], activation=None, strides=1)

            # Transitional Layer
            if stack != 0 and block == 0:
                x = resnet_layer(input_layer=x, kernel_size=1, filter_size=filter_sizes[stack], strides=2,
                                 activation=None,
                                 batch_normalization=False)

            x = add([x, y])
            x = Activation('relu')(x)
    return x


inputs = Input((32, 32, 3))
x = inputs
x = resnet_layer(input_layer=x, filter_size=16)
x = residual_stack(x, num_stack, num, filter_sizes)
x = AveragePooling2D(8)(x)
x = Flatten()(x)
outputs = Dense(units=10, activation='softmax', kernel_initializer='he_normal')(x)
model = Model(inputs=inputs, outputs=outputs)

model.summary()
plot_model(model, 'C:\\Users\\Vargha\\Desktop\\resnet_v1_20.png', show_shapes=True)


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=128, validation_data=(x_test, y_test), verbose=1)
