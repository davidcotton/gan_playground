from dataloaders.dataloader import DataLoader


class Cifar10DataLoader(DataLoader):
    def __init__(self):
        super().__init__()
        self.name = 'cifar10'

    def load_data(self):
        pass

    def next_batch(self, batch_size: int):
        pass
