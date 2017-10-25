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
from models.MNIST_1h import MNIST_1h
from variance_metric import get_activations

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
    normal_params = list(model.hidden_layer.parameters()) + list(model.output_layer.parameters())
    params = {}
    for x in normal_params:
        params[x] = {
            'weight_decay': 0.001
        } 

    optimizer = Adam(model.parameters()) 
    for e in range(0, 5):
        print("Epoch %s" % e)
        for i, (images, labels) in enumerate(dl):
            # print(round(i / len(dl) * 100))
            images = wrap(Variable(images, requires_grad=False))
            labels = wrap(Variable(labels, requires_grad=False))
            output = model(images)
            optimizer.zero_grad()
            (criterion(output, labels) + 0.01 * model.loss()).backward()
            acc = (output.max(1)[1] == labels).float().mean()
            print(acc.data.numpy()[0], model.x_0.data.numpy()[0], model.x_0.grad.data.numpy()[0])
            optimizer.step()
            model.x_0.data -= model.x_0.grad.data * 100

def get_accuracy(model, loader):
    acc = 0
    for images, labels in loader:
        images = wrap(Variable(images, volatile=True))
        labels = wrap(labels)
        predicted = model(images).data
        acc += (predicted.max(1)[1] == labels).float().mean()
    return np.array(acc) / len(loader)

def get_dl(dataset):
    return DataLoader(
        dataset(
            './datasets/%s/' % dataset.__name__,
            train=True,
            download=True,
            transform=transform),
        batch_size=128,
        pin_memory=torch.cuda.device_count() > 0,
        shuffle=True
    )


if __name__ == '__main__':
    model = MNIST_1h_flexible(1000, wrap)
