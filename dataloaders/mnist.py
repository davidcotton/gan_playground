"""Load the MNIST digit dataset."""

from dataloaders.dataloader import DataLoader
from tensorflow.examples.tutorials.mnist import input_data

HEIGHT = 28
WIDTH = 28
CHANNELS = 1


class MNISTDataLoader(DataLoader):
    def __init__(self, batch_size: int) -> None:
        super().__init__('mnist', HEIGHT, WIDTH, CHANNELS, batch_size)
        self.x_train = None
        self.y_train = None

    def load_data(self) -> None:
        """Load the dataset."""
        self.x_train = input_data.read_data_sets('./data/mnist')

    def next_batch(self):
        """Fetch the next batch of images from the dataset."""
        return self.x_train.train.next_batch(self.batch_size)[0]
