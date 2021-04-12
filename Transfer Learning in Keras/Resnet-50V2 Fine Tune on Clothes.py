import os
from tensorflow.keras import callbacks
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, Flatten, Input, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.models import Model, load_model
import numpy as np
from keras.preprocessing import image

CATEGORIES = ['hawaei', 'picture', 'plaid', 'plain', 'spotted', 'stripe', 'zigzag']

inputs = Input((224, 224, 3))
base_model = ResNet50V2(weights='imagenet', include_top=False, input_tensor=inputs)
x = Flatten()(base_model.output)
x = BatchNormalization()(x)
x = Dense(1000, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
outputs = Dense(7, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)
model.summary()

for layer in base_model.layers:
    layer.trainable = False
for i, layer in enumerate(base_model.layers):
    print(i, layer.name, "-", layer.trainable)

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_checkpoint = callbacks.ModelCheckpoint('/content/drive/MyDrive/transfer_learning.h5', monitor='val_accuracy',
                                             verbose=1, save_best_only=True)

tensorboard = callbacks.TensorBoard('/content/drive/MyDrive/transfer_learning_logs')

lr_reducer = callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                         patience=5,
                                         factor=0.1,
                                         verbose=1)

CSVlogger = callbacks.CSVLogger('/content/drive/MyDrive/training.log')

early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=25, verbose=1)

Callbacks = [model_checkpoint, lr_reducer, CSVlogger, tensorboard, early_stopping]

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = data_generator.flow_from_directory("/content/data",
                                                     target_size=(224, 224),
                                                     batch_size=64,
                                                     class_mode='categorical')
valid_generator = data_generator.flow_from_directory("/content/data",
                                                     target_size=(224, 224),
                                                     batch_size=64,
                                                     class_mode='categorical')

model_history = model.fit_generator(
    train_generator,
    steps_per_epoch=20000 // 64,
    epochs=50,
    validation_data=valid_generator,
    validation_steps=20,
    callbacks=Callbacks,
    workers=10,
    verbose=1
)

# Prediction
new_model = load_model('/content/drive/MyDrive/transfer_learning.h5')
img_path = ('/content/7129017810301_1.jpg')
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
