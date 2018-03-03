from colorama import Fore, Style
from dataloaders.dataloader import DataLoader
import keras
from keras import layers
from keras.callbacks import TensorBoard
from keras.engine.training import Model
from keras.preprocessing import image
from models.model import Model
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.python.client import timeline
import time

LATENT_DIM = 32
MODEL_NAME = 'dcgan_keras'


def write_log(callback, names, logs, batch_no):
    """Helper method to write logs to TensorBoard."""
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


class DCGANKeras(Model):
    def __init__(self, data_loader: DataLoader, save: bool = False):
        super().__init__(data_loader, save, MODEL_NAME)
        self.discriminator: Model = self.get_discriminator()
        self.generator: Model = self.get_generator()
        self.discriminator.trainable = False
        gan_input = keras.Input(shape=(LATENT_DIM,))
        gan_output = self.discriminator(self.generator(gan_input))
        self.gan = keras.models.Model(gan_input, gan_output)
        gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
        self.gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

        self.callback = TensorBoard(self.get_log_dir())
        self.callback.set_model(self.gan)
        self.train_names = ['train_loss']
        # self.validation_names = ['val_loss', 'val_mae']

    def train(self, epochs: int, d_iters=5, g_iters=1):
        batch_size = self.data_loader.batch_size
        start_time = time.time()
        for epoch in range(epochs):
            # sample random points in the latent space
            random_latent_vectors = np.random.normal(size=(batch_size, LATENT_DIM))

            # decode them to fake images
            generated_images = self.generator.predict(random_latent_vectors)

            # combine them with real images
            real_images = self.data_loader.next_batch()
            combined_images = np.concatenate([generated_images, real_images])

            # assemble labels discriminating real from fake images
            labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
            # add random noise to the labels - important trick!
            labels += 0.05 * np.random.random(labels.shape)

            # train the discriminator
            d_loss = self.discriminator.train_on_batch(combined_images, labels)

            # sample random points in the latent space
            random_latent_vectors = np.random.normal(size=(batch_size, LATENT_DIM))

            # assembles labels that say "all real images" (lie)
            misleading_targets = np.zeros((batch_size, 1))

            # train the generator
            # via the GAN model where the discriminator weights are frozen
            a_loss = self.gan.train_on_batch(random_latent_vectors, misleading_targets)
            # write_log(self.callback, self.train_names, a_loss, start_time) # @todo need to fix

            runtime = self.get_runtime(start_time)
            sys.stdout.write(
                f'\rEpoch: {Fore.GREEN}{epoch}{Style.RESET_ALL}    ' +
                f'Elapsed: {runtime}    ' +
                f'd_loss: {Fore.MAGENTA}{d_loss:.3f}{Style.RESET_ALL}    ' +
                f'a_loss: {Fore.RED}{a_loss:.3f}{Style.RESET_ALL}'
            )
            sys.stdout.flush()

            # occasionally save/plot
            if epoch % 100 == 0:
                if self.save:
                    self.gan.save_weights('gan.h5')

                # save a generated image
                img = image.array_to_img(generated_images[0] * 255, scale=False)
                img.save(os.path.join(self.get_output_dir(), 'epoch_{:06d}.png'.format(epoch)))

    def get_discriminator(self) -> Model:
        height = self.data_loader.height
        width = self.data_loader.width
        channels = self.data_loader.channels

        discriminator_input = layers.Input(shape=(height, width, channels))
        x = layers.Conv2D(128, 3)(discriminator_input)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(128, 4, strides=2)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(128, 4, strides=2)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(128, 4, strides=2)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Flatten()(x)

        # one dropout layer - important trick!
        x = layers.Dropout(0.4)(x)

        # classification layer
        x = layers.Dense(1, activation='sigmoid')(x)

        # discriminator model which turns a (32, 32, 3) input into a binary classification
        discriminator = keras.models.Model(discriminator_input, x)
        # discriminator.summary()

        # to stabilise training we use learning rate decay
        # and gradient clipping (by value) in the optimizer
        discriminator_optimizer = keras.optimizers.RMSprop(
            lr=0.0008,
            clipvalue=1.0,  # use gradient clipping
            decay=1e-8  # to stabilise training use learning rate decay
        )
        discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

        return discriminator

    def get_generator(self) -> Model:
        channels = self.data_loader.channels

        generator_input = keras.Input(shape=(LATENT_DIM,))
        # first transform the input into a 16x16 128-channel feature map
        x = layers.Dense(128 * 16 * 16)(generator_input)
        x = layers.LeakyReLU()(x)
        x = layers.Reshape((16, 16, 128))(x)

        # then add a convolution layer
        x = layers.Conv2D(256, 5, padding='same')(x)
        x = layers.LeakyReLU()(x)

        # upsample to 32x32
        x = layers.Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same')(x)
        x = layers.LeakyReLU()(x)

        # few more conv layers
        x = layers.Conv2D(256, 5, padding='same')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(256, 5, padding='same')(x)
        x = layers.LeakyReLU()(x)

        # produce a 32x32 1-channel feature map
        x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
        generator = keras.models.Model(generator_input, x)
        # generator.summary()

        return generator
