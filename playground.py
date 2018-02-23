import argparse
from datasets.cifar10 import Cifar10Dataset
from datasets.pokemon import PokemonDataset
from models.dcgan import DCGAN
from models.wgan import WGAN

DATASETS = {
    'cifar10': Cifar10Dataset(),
    'lsun-classroom': '',
    'mnist': '',
    'pokemon': PokemonDataset(),
}
MODELS = {
    'dcgan': DCGAN(),
    'wgan': WGAN(),
}


def main(args):
    data = dataset_factory(args.data)
    model = model_factory(args.model)
    print(data)
    print(model)


def dataset_factory(dataset_name):
    if dataset_name in DATASETS:
        return DATASETS[dataset_name]
    else:
        raise ValueError('Dataset does not exist "%s"' % dataset_name)


def model_factory(model_type):
    if model_type in MODELS:
        return MODELS[model_type]
    else:
        raise ValueError('Model type does not exists "%s"' % model_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--model', type=str, default='dcgan')
    # parser.add_argument('epochs', type=int, help='the number of episodes')
    # parser.add_argument('--save', help='load and save progress', action='store_true')
    # parser.add_argument('--verbose', help='verbose error logging', action='store_true')
    main(parser.parse_args())
