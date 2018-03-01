from dataloaders.dataloader import DataLoader
from keras.datasets import cifar10

HEIGHT = 32
WIDTH = 32
CHANNELS = 3


class Cifar10DataLoader(DataLoader):
    def __init__(self, batch_size: int = 20) -> None:
        super().__init__(batch_size)
        self.name = 'cifar10'
        self.x_train = None
        self.y_train = None

    def load_data(self):
        """Load the dataset."""
        (self.x_train, self.y_train), (x_test, y_test) = cifar10.load_data()
        self.x_train = self.x_train[self.y_train.flatten() == 6]  # frog images
        self.x_train = self.x_train.reshape((self.x_train.shape[0],) + (HEIGHT, WIDTH, CHANNELS))\
                           .astype('float32') / 255.0

    def next_batch(self):
        """Fetch the next batch of images from the dataset."""
        stop = self.batch_num + self.batch_size
        data = self.x_train[self.batch_num:stop]

        self.batch_num += self.batch_size
        if self.batch_num > len(self.x_train) - self.batch_size:
            self.batch_num = 0

        return data
