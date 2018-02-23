from algorithms.algorithm import Algorithm
import os
import tensorflow as tf
import time


class WGAN(Algorithm):

    def train(self, epochs: int, d_iters=5, g_iters=1):
        self.sess.run(tf.global_variables_initializer())
        start_time = time.time()

        for t in range(0, epochs):
            for d_iter in range(d_iters):
                pass

            for g_iter in range(g_iters):
                pass
