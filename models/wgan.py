from models.model import Model
import os
import tensorflow as tf
import time


class WGAN(Model):
    def train(self, epochs: int, d_iters=5, g_iters=1):
        self.sess.run(tf.global_variables_initializer())
        start_time = time.time()

        for epoch in range(epochs):
            for d_iter in range(d_iters):
                pass

            for g_iter in range(g_iters):
                pass


class Discriminator:
    def __init__(self):
        self.x_dim = 784

    def __call__(self, x):
        with tf.variable_scope(self.__class__.__name__) as scope:
            bias = tf.shape(z)[0]
            x = tf.reshape(x, [bias, 28, 28, 1])


class Generator:
    def __init__(self):
        self.z_dim = 100

    def __call__(self, z):
        with tf.variable_scope(self.__class__.__name__) as scope:
            bias = tf.shape(z)[0]
