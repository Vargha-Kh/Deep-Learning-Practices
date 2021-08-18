from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *


def get_model(num_classes, dim=(224, 224, 3), weights=None, fine_tune=False, predict=False):
    inputs = Input(dim)
    x = preprocess_input(inputs)
    base_model = ResNet152(include_top=False, input_shape=dim, pooling='avg')
    base_model.trainable = False
    x = base_model(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, x)
    if fine_tune and predict:
        for layer in model.layers:
            layer.trainable = True
    if weights is not None:
        model.load_weights(weights)
    if fine_tune and not predict:
        for layer in model.layers:
            layer.trainable = True
    return model
