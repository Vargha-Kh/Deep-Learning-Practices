import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense, concatenate, ReLU, Lambda
from tensorflow.keras import optimizers


def conv2d_bn(x, filters, kernel_size, strides, pad, name, activation='linear', group=1):
    # group = 1 or 2
    if group == 1:
        x = Conv2D(filters, kernel_size, padding=pad, strides=strides, activation=activation, name=name)(x)
    else:
        x_a, x_b = Lambda(lambda x: tf.split(x, group, axis=-1))(x)
        x_a = Conv2D(filters // 2, kernel_size, padding=pad, strides=strides, activation=activation, name=name + 'a')(
            x_a)
        x_b = Conv2D(filters // 2, kernel_size, padding=pad, strides=strides, activation=activation, name=name + 'b')(
            x_b)
        x = concatenate([x_a, x_b])
    return x


def AlexNet(img_shape=(227, 227, 3), num_classes=1000, weights='bvlc_alexnet.npy'):
    input = Input(img_shape)

    conv1 = conv2d_bn(x=input, filters=96, kernel_size=11, strides=4, pad='SAME', group=1, name='conv1')
    # conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)
    # conv1 = Lambda(lambda x: tf.nn.local_response_normalization(x, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0))(
    #     conv1)
    pool1 = MaxPooling2D(pool_size=3, strides=2)(conv1)
    conv2 = conv2d_bn(x=pool1, filters=256, kernel_size=5, strides=1, pad='SAME', group=2, name='conv2')
    conv2 = ReLU()(conv2)
    # conv2 = Lambda(
    #     lambda x: tf.nn.local_response_normalization(conv2, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0))(conv2)
    # conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=3, strides=2)(conv2)
    conv3 = conv2d_bn(x=pool2, filters=384, kernel_size=3, strides=1, pad='SAME', activation='relu', group=1,
                      name='conv3')
    conv4 = conv2d_bn(x=conv3, filters=384, kernel_size=3, strides=1, pad='SAME', activation='relu', group=2,
                      name='conv4')
    conv5 = conv2d_bn(x=conv4, filters=256, kernel_size=3, strides=1, pad='SAME', activation='relu', group=2,
                      name='conv5')
    pool5 = MaxPooling2D(pool_size=3, strides=2)(conv5)
    flatten5 = Flatten()(pool5)
    fc6 = Dense(4096, activation='relu', name='fc6')(flatten5)
    # drop6 = Dropout(0.5)(fc6)
    fc7 = Dense(4096, activation='relu', name='fc7')(fc6)
    # drop7 = Dropout(0.5)(fc7)
    fc8 = Dense(num_classes, activation='softmax', name='fc8')(fc7)

    model = Model(input, fc8)
    print('AlexNet created.')

    if weights is not None:
        weights_dic = np.load(weights, encoding='bytes', allow_pickle=True).item()
        # model.set_weights(weights_dic)
        conv1w = weights_dic["conv1"][0]
        conv1b = weights_dic["conv1"][1]
        model.get_layer('conv1').set_weights([conv1w, conv1b])

        conv2w = weights_dic["conv2"][0]
        conv2b = weights_dic["conv2"][1]
        w_a, w_b = np.split(conv2w, 2, axis=-1)
        b_a, b_b = np.split(conv2b, 2, axis=-1)
        model.get_layer('conv2a').set_weights([w_a, b_a])
        model.get_layer('conv2b').set_weights([w_b, b_b])

        conv3w = weights_dic["conv3"][0]
        conv3b = weights_dic["conv3"][1]
        model.get_layer('conv3').set_weights([conv3w, conv3b])

        conv4w = weights_dic["conv4"][0]
        conv4b = weights_dic["conv4"][1]
        w_a, w_b = np.split(conv4w, 2, axis=-1)
        b_a, b_b = np.split(conv4b, 2, axis=-1)
        model.get_layer('conv4a').set_weights([w_a, b_a])
        model.get_layer('conv4b').set_weights([w_b, b_b])

        conv5w = weights_dic["conv5"][0]
        conv5b = weights_dic["conv5"][1]
        w_a, w_b = np.split(conv5w, 2, axis=-1)
        b_a, b_b = np.split(conv5b, 2, axis=-1)
        model.get_layer('conv5a').set_weights([w_a, b_a])
        model.get_layer('conv5b').set_weights([w_b, b_b])

        fc6w = weights_dic['fc6'][0]
        fc6b = weights_dic['fc6'][1]
        model.get_layer('fc6').set_weights([fc6w, fc6b])

        fc7w = weights_dic['fc7'][0]
        fc7b = weights_dic['fc7'][1]
        model.get_layer('fc7').set_weights([fc7w, fc7b])

        fc8w = weights_dic['fc8'][0]
        fc8b = weights_dic['fc8'][1]
        model.get_layer('fc8').set_weights([fc8w, fc8b])

        print('Weights loaded.')

    return model


def get_model(n_classes, lr, weights=None):
    base_model = AlexNet(weights=weights)
    if weights is not None:
        base_model.trainable = False
        print('trainable - False')
    else:
        base_model.trainable = True
        print('trainable = True')

    model = Sequential()
    model.add(base_model)
    # model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.3))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=lr, momentum=0.9),
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    model = AlexNet(weights=None)
    print(model.summary())
