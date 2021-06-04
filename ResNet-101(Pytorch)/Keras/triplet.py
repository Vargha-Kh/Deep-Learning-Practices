import os
from collections import defaultdict
from datetime import datetime
from random import choice
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
# from tensorflow.keras.applications.resnet import preprocess_input, ResNet101
from networkx.utils import weighted_choice
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard


def get_label_list(img_dir, base_path):
    img_dir = os.path.join(base_path, img_dir)
    map_ = {class_name: en for en, class_name in enumerate(os.listdir(img_dir))}
    labels = []
    img_list = []
    for class_name in os.listdir(img_dir):
        class_dir = os.path.join(img_dir, class_name)
        label = map_[class_name]
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img_list.append(img_path)
            labels.append(label)
    return img_list, labels


def triplet_loss(y_true, y_pred, alpha=0.4):
    """
    https://github.com/KinWaiCheuk/Triplet-net-keras/blob/master/Triplet%20NN%20Test%20on%20MNIST.ipynb
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """

    total_lenght = y_pred.shape.as_list()[-1]
    anchor = y_pred[:, 0:int(total_lenght * 1 / 3)]
    positive = y_pred[:, int(total_lenght * 1 / 3):int(total_lenght * 2 / 3)]
    negative = y_pred[:, int(total_lenght * 2 / 3):int(total_lenght * 3 / 3)]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor - positive), axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor - negative), axis=1)

    # compute loss
    basic_loss = pos_dist - neg_dist + alpha
    loss = K.maximum(basic_loss, 0.0)

    return loss


def cap_sequence(seq, max_len, append):
    if len(seq) < max_len:
        if np.random.uniform(0, 1) < 0.5:
            return seq + [append] * (max_len - len(seq))
        else:
            return [append] * (max_len - len(seq)) + seq
    else:
        if np.random.uniform(0, 1) < 0.5:

            return seq[:max_len]
        else:
            return seq[-max_len:]


def cap_sequences(list_sequences, max_len, append):
    capped = []
    for seq in list_sequences:
        out_seq = cap_sequence(seq, max_len, append)
        capped.append(out_seq)

    return capped


# def read_img(path, preprocess=True):
#     img = cv2.imread(path)
#     if img is None or img.size < 10:
#         img = np.zeros((222, 171))
#     img = cv2.resize(img, (171, 222))
#     if preprocess:
#         img = preprocess_input(img)
#     return img
#

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, img_dir, base_dir, batch_size=32, dim=(224, 224), n_channels=3,
                 n_classes=1, shuffle=True, image_dir='.', val=False):
        self.dim = dim
        self.image_dir = image_dir
        self.batch_size = batch_size
        img_ids, labels = get_label_list(img_dir, base_dir)
        self.img_ids = img_ids
        self.labels = labels
        self.class_labels = defaultdict(list)
        for label, img_path in zip(self.labels, self.img_ids):
            self.class_labels[label].append(img_path)
        self.label_names = list(self.class_labels.keys())
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.indexes = None
        self.on_epoch_end()
        self.val = val

    def __len__(self):
        return int(np.floor(len(self.img_ids) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.img_ids[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp, labels)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.img_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp, labels):
        X_anchor = np.zeros((self.batch_size, *self.dim, self.n_channels))
        X_neg = np.zeros((self.batch_size, *self.dim, self.n_channels))
        X_pos = np.zeros((self.batch_size, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size), dtype=float)
        for i, (name, label) in enumerate(zip(list_IDs_temp, labels)):
            img = self.get_image(name)
            positive_path = choice(self.class_labels[label])
            pos_img = self.get_image(positive_path)
            labs = self.label_names.copy()
            labs.remove(label)
            neg_label = choice(labs)
            neg_path = choice(self.class_labels[neg_label])
            neg_img = self.get_image(neg_path)
            X_anchor[i] = img
            X_neg[i] = neg_img
            X_pos[i] = pos_img

        return [X_anchor, X_pos, X_neg], y

    def get_image(self, name):
        img = cv2.imread(name)[..., ::-1]
        # img = preprocess_input(img)
        img = self.resize_pad(img, self.dim[0])
        return img

    @staticmethod
    def resize_pad(im, desired_size=224):
        old_size = im.shape[:2]  # old_size is in (height, width) format

        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        # new_size should be in (width, height) format

        im = cv2.resize(im, (new_size[1], new_size[0]))

        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=color)

        return new_im


def image_model(lr=0.0001, weights=None, predict=None, fine_tune=None):
    input_1 = Input(shape=(None, None, 3))
    input_2 = Input(shape=(None, None, 3))
    input_3 = Input(shape=(None, None, 3))

    x1 = tf.keras.applications.resnet.preprocess_input(input_1)
    x2 = tf.keras.applications.resnet.preprocess_input(input_2)
    x3 = tf.keras.applications.resnet.preprocess_input(input_3)
    base_model = tf.keras.applications.resnet.ResNet101(weights='imagenet',
                                                        include_top=False,
                                                        pooling='avg')
    for layer in base_model.layers:
        layer.trainable = False

    x1 = base_model(x1)
    x2 = base_model(x2)
    x3 = base_model(x3)
    layer_normalizer = tf.keras.layers.LayerNormalization(name='layer_normalization')

    x1 = layer_normalizer(x1)
    x2 = layer_normalizer(x2)
    x3 = layer_normalizer(x3)

    dense_1 = Dense(vec_dim, activation="linear", name="dense_image_1", use_bias=False)

    x1 = dense_1(x1)
    x2 = dense_1(x2)
    x3 = dense_1(x3)
    _norm = Lambda(lambda x: K.l2_normalize(x, axis=-1))

    x1 = _norm(x1)
    x2 = _norm(x2)
    x3 = _norm(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    model = Model([input_1, input_2, input_3], x)

    model.compile(loss=triplet_loss, optimizer=SGD(lr, momentum=0.9))
    if fine_tune and predict:
        for layer in model.layers:
            layer.trainable = True
    if weights is not None:
        model.load_weights(weights)
        print('model has been loaded!')
    if fine_tune and not predict:
        for layer in model.layers:
            layer.trainable = True
    model.summary()

    return model


if __name__ == "__main__":
    weights = "resnet101_triplet.h5"
    file_path = 'resnet101_triplet_fine.h5'
    epochs = 50
    vec_dim = 256
    BATCH_SIZE = 16
    model = image_model(lr=0.0001, weights=weights, fine_tune=True, predict=False)
    train_get = DataGenerator(img_dir='top', base_dir='..', batch_size=BATCH_SIZE)
    valid_get = DataGenerator(img_dir='validation/top', base_dir='..', val=True, batch_size=4)

    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tbCallBack = TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=True)
    checkpointer = ModelCheckpoint(filepath=file_path, verbose=1, save_best_only=True, save_weights_only=True)
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, verbose=1)
    reduce_lr = ReduceLROnPlateau(patience=2, factor=0.1, cooldown=0, verbose=1)

    model.fit(train_get,
              use_multiprocessing=True,
              validation_data=valid_get,
              epochs=epochs,
              verbose=1,
              workers=4,
              steps_per_epoch=len(train_get),
              validation_steps=len(valid_get),
              callbacks=[tbCallBack, checkpointer, early_stopping, reduce_lr])
    model.save_weights("resnet50_triplet_last.h5")
