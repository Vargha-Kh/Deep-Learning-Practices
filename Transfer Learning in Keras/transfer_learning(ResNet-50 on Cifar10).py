import tensorflow.keras as K
import os
from tensorflow.keras import callbacks
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, Flatten, Input, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers.experimental.preprocessing import Resizing

def preprocess(x, y):
    x_p = preprocess_input(x)
    y_p = K.utils.to_categorical(y, 10)
    return x_p, y_p


(x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
x_train, y_train = preprocess(x_train, y_train)
x_test, y_test = preprocess(x_test, y_test)

inputs = Input((32, 32, 3))
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
for layer in base_model.layers[:143]:
    layer.trainable = False
for i, layer in enumerate(base_model.layers):
        print(i, layer.name, "-", layer.trainable)
# x = Resizing(height=224, width=224)(base_model.output)
x = Flatten()(base_model.output)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)
model.summary()

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_checkpoint = callbacks.ModelCheckpoint(os.path.abspath('transfer_learning.h5'), monitor='val_accuracy',
                                             verbose=1, save_best_only=True)

tensorboard = callbacks.TensorBoard(os.path.abspath('transfer_learning_logs'))

lr_reducer = callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                         patience=5,
                                         factor=0.1,
                                         verbose=1)

CSVlogger = callbacks.CSVLogger(os.path.abspath('training.log'))

early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=25, verbose=1)

Callbacks = [model_checkpoint, lr_reducer, CSVlogger, tensorboard, early_stopping]

history = model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test), callbacks=Callbacks)
