from torch.nn import CrossEntropyLoss
import re
from paper.ICML.models.FullyConnected import FullyConnected
from paper.ICML.models.VGG import VGG
from copy import deepcopy
from datasets import MNIST, FashionMNIST, CIFAR10
from torchvision import transforms
import torchsample as ts

LOG_DISTRIBUTIONS = re.compile("(" + ")|(".join([
    'lambda',
    'weight_decay',
    'learning_rate',
    'batch_size',
    'size_layer_[1-9]+',
    'classifier_layer_[1-9]+'
]) + ")")

INTEGERS = re.compile("(" + ")|(".join([
    'batch_size',
    'size_layer_[1-9]+',
    'classifier_layer_[1-9]+',
    'layers'
]) + ")")

SETTINGS = {
    'MNIST_FC_DYNAMIC': {
        'mode': 'classification',
        'model': FullyConnected,
        'dataset': MNIST,
        'val_batch_size': 1000,
        'normalization': [(0.13066047740239478,),
                          (0.30810780876661253,)],
        'data_augmentations': [],
        'params': {
            'lambda': (1e-2, 1e-6),
            'input_features': [(1, 28, 28)],
            'output_features': [10],
            'layers': (1, 5),
            'learning_rate': (1e-2, 1e-5),
            'batch_size': (8, 512),
            'dropout': [0, 0.1, 0.2, 0.5],
            'batch_norm': [True],
            'weight_decay': (1e-2, 1e-8),
            'dynamic': [True],
            'gamma': [0.99],
            'size_layer_1': [5000],
            'size_layer_2': [5000],
            'size_layer_3': [5000],
            'size_layer_4': [5000],
            'size_layer_5': [5000],
        },
    },
    'CIFAR10_VGG_DYNAMIC': {
        'mode': 'classification',
        'model': VGG,
        'dataset': CIFAR10,
        'val_batch_size': 1000,
        'normalization': [(0.4914, 0.4822, 0.4465),
                          (0.2023, 0.1994, 0.2010)],
        'data_augmentations': [
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ts.transforms.RandomRotate(10),
        ],
        'params': {
            'lambda': (1e-2, 1e-5),
            'name': ['VGG16'],
            'input_features': [(3, 32, 32)],
            'output_features': [10],
            'learning_rate': (1e-2, 1e-6),
            'batch_size': (8, 512),
            'weight_decay': (1e-2, 1e-8),
            'factor': [2],
            'classifier_layer_1': [5000],
            'classifier_layer_2': [5000],
            'gamma': [0.9, 0.99, 0],
            'batch_norm': [True],
            'dynamic': [True],
        },
    }
}

SETTINGS['MNIST_FC_STATIC'] = deepcopy(SETTINGS['MNIST_FC_DYNAMIC'])
for i in range(1, 6):
    SETTINGS['MNIST_FC_STATIC']['params']['size_layer_%s' % i] = (
        (20, SETTINGS['MNIST_FC_STATIC']['params']['size_layer_%s' % i][0])
    )
SETTINGS['MNIST_FC_STATIC']['params']['dynamic'] = [False]

SETTINGS['CIFAR10_VGG_STATIC'] = deepcopy(SETTINGS['CIFAR10_VGG_DYNAMIC'])
SETTINGS['CIFAR10_VGG_STATIC']['params']['dynamic'] = [False]
SETTINGS['CIFAR10_VGG_STATIC']['params']['classifier_layer_1'] = (32, 2500)
SETTINGS['CIFAR10_VGG_STATIC']['params']['classifier_layer_2'] = (32, 2500)
SETTINGS['CIFAR10_VGG_STATIC']['params']['factor'] = (0.1, 2)
