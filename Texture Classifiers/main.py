from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

early_stopping = EarlyStopping(monitor="loss", patience=3, verbose=1)
model_checkpoint = ModelCheckpoint('resnet50.h5', monitor='loss')

num_classes = 7
base_model = ResNet50(include_top=False, input_shape=(224, 224, 3), pooling='max')
base_model.trainable = False

input_shape = base_model.output_shape[1]

model = Sequential()
model.add(base_model)
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])

# model.summary()

print(model.summary())

generator = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                               samplewise_center=False,  # set each sample mean to 0
                               featurewise_std_normalization=False,  # divide inputs by std of the dataset
                               samplewise_std_normalization=False,  # divide each input by its std
                               zca_whitening=False,  # apply ZCA whitening
                               rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                               width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                               height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                               horizontal_flip=True,  # randomly flip images
                               vertical_flip=False  # randomly flip images
                               )

train_generator = generator.flow_from_directory("top")
model.fit_generator(train_generator,
                    steps_per_epoch=100,
                    epochs=100,
                    verbose=1,
                    workers=8,
                    callbacks=[model_checkpoint, early_stopping])
