"""Helper to assist in the loading and batching of different image datasets."""

from abc import ABC, abstractmethod
import numpy as np


class DataLoader(ABC):
    def __init__(self, name: str, height: int, width: int, channels: int, batch_size: int) -> None:
        self.name: str = name
        self.height: int = height
        self.width: int = width
        self.channels: int = channels
        self.batch_size: int = batch_size
        self.batch_num: int = 0
        self.x_train: np.ndarray = None
        self.y_train: np.ndarray = None

    @abstractmethod
    def load_data(self) -> None:
        """Load the dataset."""
        pass

    @abstractmethod
    def next_batch(self):
        """Fetch the next batch of images from the dataset."""
        pass
