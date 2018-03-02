"""Load in the Pokemon dataset, a collection of Pokemon images."""

from dataloaders.dataloader import DataLoader
import tensorflow as tf
import os

DATA_DIR = 'data/pokemon'
HEIGHT = 256
WIDTH = 256
CHANNELS = 3


class PokemonDataLoader(DataLoader):
    def __init__(self, batch_size: int) -> None:
        super().__init__('pokemon', HEIGHT, WIDTH, CHANNELS, batch_size)

    def load_data(self) -> None:
        """Load the dataset."""
        images = [os.path.join(DATA_DIR, img) for img in os.listdir(DATA_DIR)]
        all_images = tf.convert_to_tensor(images, dtype=tf.string)

    def next_batch(self):
        """Fetch the next batch of images from the dataset."""
        pass
