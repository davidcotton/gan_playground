from abc import ABC, abstractmethod
from colorama import Fore, Style
from dataloaders.dataloader import DataLoader
from datetime import timedelta
import os
import tensorflow as tf
import time

BASE_OUT_DIR = 'out'
BASE_LOG_DIR = 'log'


class Model(ABC):
    def __init__(self, data_loader: DataLoader, save_model: bool, name: str = None):
        self.data_loader = data_loader
        self.sess = tf.Session()
        self.name: str = name
        self.save_model = save_model

    @abstractmethod
    def train(self, epochs: int, d_iters=5, g_iters=1):
        """Train the model."""
        pass

    def get_output_dir(self):
        """Get the name of the model's output directory."""
        model_dir = '{}/{}'.format(self.get_base_output_dir(), self.get_name())
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        dataset_dir = '{}/{}'.format(model_dir, self.data_loader.name)
        if not os.path.isdir(dataset_dir):
            os.makedirs(dataset_dir)

        return dataset_dir

    @staticmethod
    def get_base_output_dir():
        """Get the base output directory."""
        return BASE_OUT_DIR

    def get_log_dir(self):
        """Get the name of the model's logging directory."""
        model_dir = '{}/{}'.format(self.get_base_output_dir(), self.get_name())
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        return model_dir

    @staticmethod
    def get_base_log_dir():
        """Get the base logging directory."""
        return BASE_LOG_DIR

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
        return f'{Fore.YELLOW}{hours}{Style.RESET_ALL} hours ' + \
               f'{Fore.YELLOW}{mins:02d}{Style.RESET_ALL} mins ' + \
               f'{Fore.YELLOW}{seconds:02d}{Style.RESET_ALL} seconds'


def leaky_relu(x, n, leak=0.2):
    """A simple leaky RELU implementation.
    @todo replace with tf.nn.relu (since tf 1.4)
    """
    return tf.maximum(x, leak * x, name=n)
