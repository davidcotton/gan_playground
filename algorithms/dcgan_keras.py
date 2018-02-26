from algorithms.algorithm import Algorithm
from dataloaders.dataloader import DataLoader
import keras
from keras import layers
from keras.engine.training import Model
from keras.preprocessing import image
import numpy as np
import os
import time

LATENT_DIM = 32
HEIGHT = 32
WIDTH = 32
CHANNELS = 3
NAME = 'dcgan_keras'


class DCGANKeras(Algorithm):
    def __init__(self, data_loader: DataLoader, save_file_name: str = None):
        super().__init__(data_loader, save_file_name, NAME)
        self.discriminator: Model = self.get_discriminator()
        self.generator: Model = self.get_generator()
        self.discriminator.trainable = False
        gan_input = keras.Input(shape=(LATENT_DIM,))
        gan_output = self.discriminator(self.generator(gan_input))
        self.gan = keras.models.Model(gan_input, gan_output)
        gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
        self.gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

    def train(self, epochs: int, d_iters=5, g_iters=1):
        (x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
        x_train = x_train[y_train.flatten() == 6]  # frog images
        x_train = x_train.reshape((x_train.shape[0],) + (HEIGHT, WIDTH, CHANNELS)).astype('float32') / 255.
        batch_size = 20

        start = 0
        start_time = time.time()
        for epoch in range(epochs):
            # sample random points in the latent space
            random_latent_vectors = np.random.normal(size=(batch_size, LATENT_DIM))

            # decode them to fake images
            generated_images = self.generator.predict(random_latent_vectors)

            # combine them with real images
            stop = start + batch_size
            real_images = x_train[start:stop]
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

            start += batch_size
            if start > len(x_train) - batch_size:
                start = 0

            # occasionally save/plot
            if epoch % 1 == 0:
                # save model weights
                # self.gan.save_weights('gan.h5')

                # print metrics
                print('  discriminator loss at epoch %s: %s' % (epoch, d_loss))
                print('  adversarial loss at epoch %s: %s' % (epoch, a_loss))
                print('  elapsed runtime: %s' % self.get_runtime(start_time))
                print()

                # save a generated image
                # img = image.array_to_img(generated_images[0] * 255, scale=False)
                # img.save(os.path.join(self.get_out_dir(), 'epoch_{:06d}.png'.format(epoch)))

        # print('total runtime: %0.3f' % self.get_runtime(start_time))

    def get_discriminator(self) -> Model:
        discriminator_input = layers.Input(shape=(HEIGHT, WIDTH, CHANNELS))
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
        x = layers.Conv2D(CHANNELS, 7, activation='tanh', padding='same')(x)
        generator = keras.models.Model(generator_input, x)
        # generator.summary()

        return generator

    def get_out_dir(self):
        model_name = self.get_name()
        model_dir = '{}/{}'.format(self.get_base_out_dir(), model_name)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        dataset_name = self.data_loader.get_name()
        dataset_dir = '{}/{}'.format(model_dir, dataset_name)
        if not os.path.isdir(dataset_dir):
            os.makedirs(dataset_dir)

        return dataset_dir
