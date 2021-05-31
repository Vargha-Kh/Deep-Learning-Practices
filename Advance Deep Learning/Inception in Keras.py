from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, AveragePooling2D, Flatten, Dropout, \
    GlobalAveragePooling2D, Concatenate, Dense
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import concatenate

DATADIR = '/home/vargha/Desktop/top'
path = '/home/vargha/Desktop/top'
target = '/home/vargha/Desktop/top'
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


def inception_module(input_layer, f1, f2_conv1, f2_conv3, f3_conv1, f3_conv5, f4):
    conv1 = Conv2D(filters=f1, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    conv2 = Conv2D(filters=f2_conv1, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=f2_conv3, kernel_size=(3, 3), padding='same', activation='relu')(conv2)

    conv3 = Conv2D(filters=f3_conv1, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    conv3 = Conv2D(filters=f3_conv5, kernel_size=(5, 5), padding='same', activation='relu')(conv3)

    conv4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_layer)
    conv4 = Conv2D(filters=f4, kernel_size=(1, 1), padding='same', activation='relu')(conv4)

    output_layer = concatenate([conv1, conv2, conv3, conv4], axis=-1)

    return output_layer


def GoogleNet():
    input_layer = Input((224, 224, 3))

    x = Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='valid', activation='relu')(input_layer)
    x = MaxPooling2D((3, 3), strides=2)(x)
    x = Conv2D(filters=64, kernel_size=(1, 1), strides=2, padding='valid', activation='relu')(x)
    x = Conv2D(filters=192, kernel_size=(3, 3), strides=1, padding='valid', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=2)(x)
    x = inception_module(x, 64, 96, 128, 16, 32, 32)
    x = inception_module(x, 128, 128, 192, 32, 96, 64)
    x = MaxPooling2D((3, 3), strides=2)(x)
    x = inception_module(x, 192, 96, 208, 16, 48, 64)

    x1 = AveragePooling2D((5, 5), strides=3)(x)
    x1 = Conv2D(128, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x1)
    x1 = Flatten()(x1)
    x1 = Dense(1024, activation='relu')(x1)
    x1 = Dropout(0.7)(x1)
    x1 = Dense(7, activation='softmax')(x1)

    x = inception_module(x, 160, 112, 221, 24, 64, 64)
    x = inception_module(x, 128, 128, 256, 24, 64, 64)
    x = inception_module(x, 112, 144, 288, 32, 64, 64)

    x2 = AveragePooling2D((5, 5), strides=3)(x)
    x2 = Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x2)
    x2 = Flatten()(x2)
    x2 = Dense(1024, activation='relu')(x2)
    x2 = Dropout(0.7)(x2)
    x2 = Dense(7, activation='softmax')(x2)

    x = inception_module(x, 256, 160, 620, 62, 128, 128)
    x = MaxPooling2D((3, 3), strides=2)(x)
    x = inception_module(x, 256, 160, 320, 32, 128, 128)
    x = inception_module(x, 384, 192, 384, 48, 128, 128)
    x = GlobalAveragePooling2D(name='GlobalAveragePooling2D')(x)
    x = Dropout(0.4)(x)
    x = Dense(1000)(x)
    x = Dense(7, activation='softmax', kernel_initializer='he_normal')(x)

    model_google = Model(input_layer, [x, x1, x2], name='GoogleNet')
    return model_google


model = GoogleNet()
model.summary()

model_checkpoint = callbacks.ModelCheckpoint('/home/vargha/Desktop/callbacks/model_checkpoint.h5',
                                             monitor='val_accuracy',
                                             verbose=1, save_best_only=True)

tensorboard = callbacks.TensorBoard('/home/vargha/Desktop/callbacks/logs1')

lr_reducer = callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                         patience=5,
                                         factor=0.1,
                                         verbose=1)

CSVlogger = callbacks.CSVLogger('/home/vargha/Desktop/callbacks/training.log')

early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=35, verbose=1)

Callbacks = [model_checkpoint, lr_reducer, CSVlogger, tensorboard, early_stopping]

model.compile(loss="categorical_crossentropy", optimizer=Adam(0.001), metrics=["accuracy"])

model.fit(train, batch_size=32, epochs=400, validation_data=val, verbose=1)
