# GAN Playground
A playground to test the performance of Generative Adversarial Networks

## Installation
1. Create a virtualenv called `venv`
    
        virtualenv venv

1. Activate the virtualenv

        source venv/bin/activate
        
1. Install dependencies

        pip3 install -r requirements.txt


## Usage
1. Activate the virtualenv

        source venv/bin/activate
        
1. Run the playground

        python3 playground.py --options


### Options
| What | Argument | Description | Optional | Default Value |
| ---- | -------- | ----------- | -------- | ------------- |
| Dataset | --dataset=pokemon | Which of the included datasets to load | Y | MNIST |
| Algorithm | --algorithm=dcgan | Which of the included algorithms to use | Y | WGAN |
| Epochs | --epochs=1000 | How many epochs to train for | Y | 100000 |
| Save | --save | Whether to save/checkpoint training progress | Y | No |
| Verbose | --verbose | Whether to log debugging info or not | Y | No |


### Requirements
- Python 3

## References
- https://github.com/llSourcell/Pokemon_GAN
- https://github.com/moxiegushi/pokeGAN
- https://github.com/jiamings/wgan
- https://arxiv.org/abs/1406.2661
- https://arxiv.org/abs/1701.07875
- https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/8.5-introduction-to-gans.ipynb
