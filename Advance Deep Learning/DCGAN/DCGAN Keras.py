import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Reshape, LeakyReLU, Activation, Dense, Input, Conv2D, Conv2DTranspose, \
    BatchNormalization, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import math
from tensorflow.keras.callbacks import TensorBoard

log_dir = os.path.abspath('/log')


def build_generator(inputs, image_size):
    image_resize = image_size // 4
    kernel_size = 5
    layers_filters = [128, 64, 32, 1]

    x = Dense(image_resize * image_resize * layers_filters[0])(inputs)
    x = Reshape((image_resize, image_resize, layers_filters[0]))(x)
    for filters in layers_filters:
        if filters > layers_filters[-2]:
            strides = 2
        else:
            strides = 1
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(filters, kernel_size, strides, padding='same')(x)
    x = Activation('sigmoid')(x)
    generator = Model(inputs, x, name='generator')
    return generator


def build_discriminator(generated_img):
    layer_filters = [32, 64, 128, 256]
    kernel_size = 5
    x = generated_img
    for filters in layer_filters:
        if filters == layer_filters[3]:
            strides = 1
        else:
            strides = 2
        x = LeakyReLU(0.2)(x)
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)
    discriminator = Model(generated_img, x, name='discriminator')
    return discriminator


def plot_images(generator, noise_input, show=True, step=0):
    model_name = 'DCGAN'
    os.makedirs(model_name, exist_ok=True)
    filename = os.path.join(model_name, "%05d.png" % step)
    images = generator.predict(noise_input)
    plt.figure(figsize=(6, 6))
    num_images = images.shape[0]
    image_size = images.shape[1]
    rows = int(math.sqrt(noise_input.shape[0]))
    for i in range(num_images):
        plt.subplot(rows, rows, i + 1)
        image = np.reshape(images[i], [image_size, image_size])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.savefig(filename)
    if show:
        plt.show()
    else:
        plt.close('all')


def train(models, x_train, params):
    test_size = 16
    save_interval = 500
    generator, discriminator, adversarial = models
    batch_size, latent_dim, train_steps = params
    train_size = x_train.shape[0]

    noise_inputs = np.random.uniform(-1, 1, size=[test_size, latent_dim])
    for step in range(1, train_steps + 1):
        random_batches = np.random.randint(0, train_size, size=batch_size)
        noise_z = np.random.uniform(-1, 1, size=[batch_size, latent_dim])

        real_images = x_train[random_batches]
        fake_images = generator.predict(noise_z)
        discriminator_x = np.concatenate((real_images, fake_images))

        real_labels = np.ones([batch_size, 1])
        fake_labels = np.zeros([batch_size, 1])
        discriminator_y = np.concatenate((real_labels, fake_labels))

        loss_discriminator, acc_discriminator = discriminator.train_on_batch(discriminator_x, discriminator_y)
        log_discriminator = "%d: [discriminator loss: %f, acc: %f]" % (step, loss_discriminator, acc_discriminator)

        adversarial_x = np.random.uniform(-1, 1, size=[batch_size, latent_dim])
        adversarial_y = np.ones((batch_size, 1))
        loss_adversarial, acc_adversarial = adversarial.train_on_batch(adversarial_x, adversarial_y)
        log_adversarial = "%s [adversarial loss: %f, acc: %f]" % (log_discriminator, loss_adversarial, acc_adversarial)

        TensorBoard(log_dir=log_dir)
        print(log_adversarial)
        if (step + 1) % save_interval == 0:
            if (step + 1) == train_steps:
                show = True
            else:
                show = False
            plot_images(generator, noise_inputs, show=show, step=(step + 1))
    generator.save("DCGAN" + ".h5")


def preprocess_model_and_train():
    latent_dim = 100
    batch_size = 128
    train_steps = 40000
    lr = 2e-4
    decay = 6e-8

    (x_train, _), (x_test, _) = mnist.load_data()
    channel_size = 1
    image_size = x_train.shape[1]
    input_shape = (image_size, image_size, channel_size)
    x_train = x_train.reshape((-1, image_size, image_size, channel_size)).astype('float32') / 255

    dis_inputs = Input(shape=input_shape)
    discriminator = build_discriminator(dis_inputs)
    optimizer = Adam(lr=lr, decay=decay)
    discriminator.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    input_shape = (latent_dim,)
    gen_inputs = Input(shape=input_shape)
    generator = build_generator(gen_inputs, image_size)
    optimizer = Adam(lr=lr * 0.5, decay=decay)

    discriminator.trainable = False
    adversarial = Model(gen_inputs, discriminator(generator(gen_inputs)))
    adversarial.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    models = (generator, discriminator, adversarial)
    params = (batch_size, latent_dim, train_steps)
    train(models, x_train, params)


def test_generator(generator):
    noise_inputs = np.random.uniform(-1.0, 1.0, size=[16, 100])
    plot_images(generator, noise_inputs)


if __name__ == '__main__':
    preprocess_model_and_train()
