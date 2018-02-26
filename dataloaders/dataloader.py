from abc import ABC, abstractmethod


class DataLoader(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.name = None

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def next_batch(self, batch_size: int):
        pass

    def get_name(self):
        if self.name is None:
            raise NotImplementedError('Data loader name not implemented in %s' % self.__class__.__name__)
        return self.name
