from models.model import Model
import os
import tensorflow as tf
import time


class DCGAN(Model):

    def train(self, epochs: int, d_iters=5, g_iters=1):
        self.sess.run(tf.global_variables_initializer())
        start_time = time.time()
