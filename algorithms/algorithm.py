from abc import ABC, abstractmethod
from dataloaders.dataloader import DataLoader
import os
import tensorflow as tf

OUT_DIR = 'out'


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
        pass

    @abstractmethod
    def out_dir(self):
        pass

    def get_name(self):
        if self.name is None:
            raise NotImplementedError('Data loader name not implemented in %s' % self.__class__.__name__)
        return self.name

    @staticmethod
    def get_out_dir():
        return OUT_DIR


def leaky_relu(x, n, leak=0.2):
    return tf.maximum(x, leak * x, name=n)
