from abc import ABC, abstractmethod


class DataLoader(ABC):
    def __init__(self, batch_size: int = 0) -> None:
        super().__init__()
        self.name = None
        self.batch_size = batch_size
        self.batch_num = 0

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def next_batch(self):
        pass

    def get_name(self):
        if self.name is None:
            raise NotImplementedError('Data loader name not implemented in %s' % self.__class__.__name__)
        return self.name
