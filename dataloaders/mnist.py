"""Load the MNIST digit dataset."""

from dataloaders.dataloader import DataLoader
from tensorflow.examples.tutorials.mnist import input_data


class MNISTDataLoader(DataLoader):
    def __init__(self):
        super().__init__()
        self.shape = [28, 28, 1]
        self.data = self.load_data()
        self.current_batch_num = 0
        self.name = 'mnist'

    def load_data(self):
        return input_data.read_data_sets('./data/mnist')

    def next_batch(self, batch_size: int):
        return self.data.train.next_batch(batch_size)[0]
