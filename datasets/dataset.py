from abc import ABC, abstractmethod


class Dataset(ABC):

    @abstractmethod
    def extract(self):
        pass
