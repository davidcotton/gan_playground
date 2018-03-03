import argparse
import colorama
from dataloaders.dataloader import DataLoader
from dataloaders.cifar10 import Cifar10DataLoader
from dataloaders.imagenet import ImagenetDataLoader
from dataloaders.mnist import MNISTDataLoader
from dataloaders.pokemon import PokemonDataLoader
from models.model import Model
from models.dcgan import DCGAN
from models.dcgan_keras import DCGANKeras
from models.wgan import WGAN

DATASETS = {
    'cifar10': Cifar10DataLoader,
    'imagenet': ImagenetDataLoader,
    # 'lsun-classroom': '',
    'mnist': MNISTDataLoader,
    'pokemon': PokemonDataLoader,
}
MODELS = {
    'dcgan': DCGAN,
    'dcgan_keras': DCGANKeras,
    'wgan': WGAN,
}
SAVE_FILE_NAME = 'checkpoint/checkpoint.ckpt'
BATCH_SIZE = 32


def main(args):
    data_loader = dataset_factory(args.dataset)
    model = model_factory(args.model, data_loader, args.save)
    model.train(args.epochs)


def dataset_factory(dataset_name: str):
    if dataset_name in DATASETS:
        data_loader = DATASETS[dataset_name](BATCH_SIZE)
        data_loader.load_data()
        return data_loader
    else:
        raise ValueError('Dataset does not exist "%s"' % dataset_name)


def model_factory(model_type: str, data_loader: DataLoader, save: bool) -> Model:
    if model_type in MODELS:
        model = MODELS[model_type]
        return model(data_loader, save)
    else:
        raise ValueError('Model type does not exists "%s"' % model_type)


if __name__ == '__main__':
    colorama.init()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', help='which dataset to load')
    parser.add_argument('--model', type=str, default='wgan', help='which GAN algorithm to use')
    parser.add_argument('--epochs', type=int, default=100000, help='the number of episodes')
    parser.add_argument('--save', help='load and save progress', action='store_true')
    parser.add_argument('--verbose', help='verbose error logging', action='store_true')
    main(parser.parse_args())
