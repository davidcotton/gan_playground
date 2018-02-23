from abc import ABC, abstractmethod
from dataloaders.dataloader import DataLoader
import os
import tensorflow as tf


class Algorithm(ABC):

    def __init__(self, data_loader: DataLoader, save_file_name: str=None):
        self.data_loader = data_loader
        self.sess = tf.Session()
        # self.saver = tf.train.Saver()
        # if save_file_name is not None:
        #     if os.path.isfile(save_file_name):
        #         self.saver.restore(self.sess, save_file_name)
        derp = 1

    @abstractmethod
    def train(self, epochs: int, d_iters=5, g_iters=1):
        pass
