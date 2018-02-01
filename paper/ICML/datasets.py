from torchvision import transforms
from torchvision.datasets import (
    MNIST as MNISTLoader,
    FashionMNIST as FashionMNISTLoader,
    CIFAR10 as CIFAR10Loader,
    CIFAR100 as CIFAR100Loader
)

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
