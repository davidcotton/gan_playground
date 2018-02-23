from abc import ABC, abstractmethod


class DataLoader(ABC):

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def next_batch(self, batch_size: int):
        pass
