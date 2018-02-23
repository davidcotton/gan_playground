import argparse
from algorithms.dcgan import DCGAN
from algorithms.wgan import WGAN
from dataloaders.dataloader import DataLoader
from dataloaders.cifar10 import Cifar10DataLoader
from dataloaders.mnist import MNISTDataLoader
from dataloaders.pokemon import PokemonDataLoader

DATASETS = {
    'cifar10': Cifar10DataLoader,
    # 'lsun-classroom': '',
    'mnist': MNISTDataLoader,
    'pokemon': PokemonDataLoader,
}
ALGORITHMS = {
    'dcgan': DCGAN,
    'wgan': WGAN,
}


def main(args):
    data_loader = dataset_factory(args.data)
    algorithm = algorithm_factory(args.algorithm, data_loader)
    batch_size = 1
    # data_loader.next_batch(batch_size)
    algorithm.train(args.epochs)


def dataset_factory(dataset_name: str):
    if dataset_name in DATASETS:
        return DATASETS[dataset_name]()
    else:
        raise ValueError('Dataset does not exist "%s"' % dataset_name)


def algorithm_factory(algorithm_type: str, data_loader: DataLoader):
    if algorithm_type in ALGORITHMS:
        return ALGORITHMS[algorithm_type](data_loader)
    else:
        raise ValueError('Model type does not exists "%s"' % algorithm_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', help='which dataset to load')
    parser.add_argument('--algorithm', type=str, default='wgan', help='which GAN algorithm to use')
    parser.add_argument('--epochs', type=int, default=10, help='the number of episodes')
    # parser.add_argument('--save', help='load and save progress', action='store_true')
    # parser.add_argument('--verbose', help='verbose error logging', action='store_true')
    main(parser.parse_args())
