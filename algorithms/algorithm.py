from abc import ABC, abstractmethod
from dataloaders.dataloader import DataLoader
from datetime import timedelta
import os
import tensorflow as tf
import time

BASE_OUT_DIR = 'out'


class Algorithm(ABC):
    def __init__(self, data_loader: DataLoader, save_file_name: str = None, name: str = None):
        self.data_loader = data_loader
        self.sess = tf.Session()
        # self.saver = tf.train.Saver()
        # if save_file_name is not None:
        #     if os.path.isfile(save_file_name):
        #         self.saver.restore(self.sess, save_file_name)
        self.name: str = name

    @abstractmethod
    def train(self, epochs: int, d_iters=5, g_iters=1):
        """Train the model."""
        pass

    @abstractmethod
    def get_out_dir(self):
        """Get the name of the algorithm's output directory."""
        pass

    @staticmethod
    def get_base_out_dir():
        """Get the base output directory."""
        return BASE_OUT_DIR

    def get_name(self):
        """Get the name of the algorithm class."""
        if self.name is None:
            raise NotImplementedError('Data loader name not implemented in %s' % self.__class__.__name__)
        return self.name

    @staticmethod
    def get_runtime(start_time):
        """Get the elapsed runtime."""
        diff = time.time() - start_time
        elapsed = timedelta(seconds=diff)
        hours = elapsed.seconds // 3600
        mins = (elapsed.seconds // 60) % 60
        seconds = elapsed.seconds % 60
        return '{} hours, {} mins, {} seconds'.format(hours, mins, seconds)


def leaky_relu(x, n, leak=0.2):
    """A simple leaky RELU implementation.
    @todo replace with tf.nn.relu (since tf 1.4)
    """
    return tf.maximum(x, leak * x, name=n)
