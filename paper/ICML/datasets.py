import torch
import numpy as np
from torchvision import transforms
from sklearn import preprocessing
from torch.utils.data import TensorDataset
from torchvision.datasets import (
    MNIST as MNISTLoader,
    FashionMNIST as FashionMNISTLoader,
    CIFAR10 as CIFAR10Loader,
    CIFAR100 as CIFAR100Loader
)
from sklearn.model_selection import train_test_split
from linear_regression_experiment import normalize
import openml


def load_dataset(id, do_normalize=True, factor=1):
    dataset = openml.datasets.get_dataset(id)
    X, y, categorical = dataset.get_data(
        target=dataset.default_target_attribute,
        return_categorical_indicator=True)
    lines_without_nans = (np.isnan(X).sum(1) == 0)
    X = X[lines_without_nans]
    y = y[lines_without_nans]
    if do_normalize:
        X = normalize(X)
    if any(categorical):
        enc = preprocessing.OneHotEncoder(categorical_features=categorical)
        X = enc.fit_transform(X).todense()
    return X, y


def MNIST(train=True, transform=[]):
    total_transform = transforms.Compose([transforms.ToTensor()] + transform)
    return MNISTLoader('../../datasets', download=True, train=train,
                       transform=total_transform)

def FashionMNIST(train=True, transform=[]):
    total_transform = transforms.Compose([transforms.ToTensor()] + transform)
    return FashionMNISTLoader('../../datasets',
                       download=True,
                       train=train,
                       transform=total_transform
                       )

def CIFAR10(train=True, transform=[]):
    total_transform = transforms.Compose([transforms.ToTensor()] + transform)
    return CIFAR10Loader('../../datasets',
                       download=True,
                       train=train,
                       transform=total_transform
                       )

def CIFAR100(train=True, transform=[]):
    total_transform = transforms.Compose([transforms.ToTensor()] + transform)
    return CIFAR100Loader('../../datasets',
                       download=True,
                       train=train,
                       transform=total_transform
                       )

def covertype(train=True, test_split=0.1, seed=0, normalized=True, transform=[]):
    X, y = load_dataset(1596)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=seed)
    if train:
        return TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    else:
        return TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
