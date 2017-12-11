import numpy as np
import torch
from itertools import product
from torchvision.datasets import MNIST, FashionMNIST
from utils.MNIST import get_dl as get_MNIST
from utils.Add10 import Add10Dataset, get_dl as get_Add10
from utils.Airfoil import AirfoilDataset, get_dl as get_Airfoil
from utils.Poker import PokerDataset, get_dl as get_Poker, weights as poker_weights
from utils.misc import PreloadedDataloader
from models.MultiLayerDynamicPerceptron import MultiLayerDynamicPerceptron
from models.DynamicCNN import DynamicCNN
from uuid import uuid4
from sys import argv

from algorithms.block_sparse_training_dynamic import compress_train, grow

LAMBDAS = 10.0**-np.arange(1, 7)
DECAYS = [1.0]
LAYERS = [int(argv[-2]) if len(argv) > 2 else 1]
SIZES_PER_LAYER = {
    1: 10000,
    2: 6000,
    3: 4000,
    4: 2500
}
SIZES_PER_LAYER_CNN = {
    1: 100,
    2: 75,
    3: 65,
    4: 25
}
ITERATIONS = range(5)
DATASETS = {
    'MNIST': {
        'features_in': 28*28,
        'features_out': 10,
        'get_train_dl': lambda: get_MNIST(MNIST),
        'get_test_dl': lambda: get_MNIST(MNIST, False),
        'mode': 'classification',
        'lambda_shift': 1,
        'reference': 0.984
    },
    'FashionMNIST': {
        'features_in': 28*28,
        'features_out': 10,
        'get_train_dl': lambda: get_MNIST(FashionMNIST),
        'get_test_dl': lambda: get_MNIST(FashionMNIST, False),
        'mode': 'classification',
        'lambda_shift': 1
    },
    'Add10': {
        'features_in': 10,
        'features_out': 1,
        'get_train_dl': lambda: get_Add10(Add10Dataset()),
        'get_test_dl': lambda: get_Add10(Add10Dataset(train=False), train=False),
        'mode': 'regression',
        'lambda_shift': 1e4
    },
    'Airfoil': {
        'features_in': 5,
        'features_out': 1,
        'get_train_dl': lambda: get_Airfoil(AirfoilDataset()),
        'get_test_dl': lambda: get_Airfoil(AirfoilDataset(train=False), train=False),
        'mode': 'regression',
        'lambda_shift': 1e4,
        'reference': 12.34
    },
    'Poker': {
        'features_in': 85,
        'features_out': 10,
        'get_train_dl': lambda: get_Poker(PokerDataset()),
        'get_test_dl': lambda: get_Poker(PokerDataset(train=False), train=False),
        'weights': poker_weights,
        'mode': 'classification',
        'lambda_shift': 0.1
    },
    'CNNMNIST': {
        'features_in': (1, 28, 28),
        'features_out': 10,
        'get_train_dl': lambda: get_MNIST(MNIST, True),
        'get_test_dl': lambda: get_MNIST(MNIST, False),
        'mode': 'classification',
        'lambda_shift': 1e-1,
        'reference': 0.9905,
        'time': 15
    },
}

if __name__ == "__main__":
    DS = 'MNIST'
    if argv[-1] in DATASETS.keys():
        DS = argv[-1]
    print(DS)
    train_set, val_set = PreloadedDataloader(DATASETS[DS]['get_train_dl']()).split(0.9)
    test_set = PreloadedDataloader(DATASETS[DS]['get_test_dl']())
    path_template = "./experiments/%s/%s.experiment"
    LAMBDAS *= DATASETS[DS]['lambda_shift']
    param_generator = list(product(LAMBDAS, DECAYS, LAYERS, ITERATIONS))
    print(len(param_generator))
    for params in param_generator:
        lamb, lamb_deca, layers, _ = params
        f_in = DATASETS[DS]['features_in']
        f_out = DATASETS[DS]['features_out']
        if hasattr(f_in, '__len__'): # This is a tuple, so a picture => CNN
            initial_size = SIZES_PER_LAYER_CNN[layers]
        else:
            initial_size = SIZES_PER_LAYER[layers]
        print(params, initial_size)
        id = uuid4()
        filename = path_template % (DS, id)
        if hasattr(f_in, '__len__'): # This is a tuple, so a picture => CNN
            model = DynamicCNN(
                layers,
                in_features=f_in,
                out_features=f_out,
                conv_initial_size=initial_size
            )
        else:
            model = MultiLayerDynamicPerceptron(
                layers,
                in_features=f_in,
                out_features=f_out,
                initial_size=initial_size
            )
        model = model.cuda()
        grow(model)
        time = DATASETS[DS].get('time', 5)
        stats = compress_train(model, train_set, val_set,
            test_set, lamb, lamb_deca, 0, time, mode = DATASETS[DS]['mode'],
            weight=DATASETS[DS].get('weights', None)
            )
        logs = stats.logs
        summary = (params, logs)
        torch.save(summary, open(filename, 'wb'))
