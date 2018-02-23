from algorithms.algorithm import Algorithm
import tensorflow as tf
import time


class WGAN(Algorithm):

    def train(self, epochs: int):
        # self.sess.run(tf.global_variables_initializer())
        start_time = time.time()

        for t in range(0, epochs):

    def test(self):
        pass
