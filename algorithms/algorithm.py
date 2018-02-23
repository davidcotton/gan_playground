from abc import ABC, abstractmethod
from dataloaders.dataloader import DataLoader


class Algorithm(ABC):

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    @abstractmethod
    def train(self, epochs: int):
        pass

    @abstractmethod
    def test(self):
        pass
