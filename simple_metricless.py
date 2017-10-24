import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scipy
from algorithms.exp_norm_mixture_fit import fit as fit_exp_norm
from algorithms.digamma_mixture_fit import fit as fit_digamma

from models.MNIST_1h_flexible import MNIST_1h_flexible

REPLICATES = 11

transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
])

if torch.cuda.device_count() > 0:
    wrap = lambda x: x.cuda(async=True) if torch.is_tensor(x) and x.is_pinned() else x.cuda()
    unwrap = lambda x: x.cpu()
else:
    wrap = lambda x: x
    unwrap = wrap

def train(model, dl):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())

    for e in range(0, 30):
        print("Epoch %s" % e)
        for i, (images, labels) in enumerate(dl):
            print(round(i / len(dl) * 100))
            images = wrap(Variable(images, requires_grad=False))
            labels = wrap(Variable(labels, requires_grad=False))
            output = model(images)
            loss = criterion(output, labels) + 0.01 * model.loss()
            loss.backward()
            acc = (output.max(1)[1] == labels).float().mean()
            res = (acc, model.x_0, model.k)
            return res
            optimizer.step()
            optimizer.zero_grad()
            return res

def get_accuracy(model, loader):
    acc = 0
    for images, labels in loader:
        images = wrap(Variable(images, volatile=True))
        labels = wrap(labels)
        predicted = model(images).data
        acc += (predicted.max(1)[1] == labels).float().mean()
    return np.array(acc) / len(loader)

def get_dl(dataset, prefix):
    return DataLoader(
        dataset(
            './datasets/%s/' % prefix,
            train=True,
            download=True,
            transform=transform),
        batch_size=128,
        pin_memory=torch.cuda.device_count() > 0,
        shuffle=True
    )

if __name__ == '__main__':
    model = MNIST_1h_flexible(500, wrap)

