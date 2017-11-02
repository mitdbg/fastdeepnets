from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.multiprocessing import Pool
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
from models.MNIST_1h_flexible_sorted import MNIST_1h_flexible_sorted
from models.MNIST_1h_flexible_scaled import MNIST_1h_flexible_scaled
from models.MNIST_1h import MNIST_1h
from variance_metric import get_activations, train as simple_train
from simple_metricless import get_dl, get_accuracy
from modules.SemiFlexibleLinear import SemiFlexibleLinear

class MultiLayerFlexibleMNIST(nn.Module):

    def __init__(self, hidden_layers, max_size):
        super(MultiLayerFlexibleMNIST, self).__init__()
        layers = []
        layers.append(SemiFlexibleLinear(28*28, max_size, original_size=25).cuda())
        layers.append(nn.ReLU())
        for _ in range(hidden_layers - 1):
            layers.append(SemiFlexibleLinear(max_size, max_size, original_size=25).cuda())
            layers.append(nn.ReLU())
        layers.append(nn.Linear(max_size, 10))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x.view(-1, 28*28))

    def capacity(self):
        cap = None
        for m in self.modules():
            if isinstance(m, SemiFlexibleLinear):
                current_capacity = m.capacity()
                if cap is None:
                    cap = current_capacity
                else:
                    cap += current_capacity
        return cap
    
    def size_parameters(self):
        result = set()
        for m in self.modules():
            if isinstance(m, SemiFlexibleLinear):
                result.add(m.size)
        return result

EPOCHS = 15

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

def train(models, dl, lamb=0.001, epochs=EPOCHS, l2_penalty=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizers = []
    for model in models:
        normal_params = set(model.parameters())
        size_params = model.size_parameters()
        for x in size_params:
            normal_params.remove(x)

        optimizer = Adam([{
            'params': normal_params,
            'weight_decay': l2_penalty,
        }, {
            'params': size_params,
            'lr': 1,
        }])
        optimizers.append(optimizer)
    for e in range(0, epochs):
        print("Epoch %s" % e)
        for i, (images, labels) in enumerate(dl):
            images = wrap(Variable(images, requires_grad=False))
            labels = wrap(Variable(labels, requires_grad=False))
            for model, optimizer in zip(models, optimizers):
                output = model(images)
                acc = (output.max(1)[1] == labels).float().mean()
                def tn(x):
                    return x.cpu().numpy()[0]
                print(tn(acc.data), torch.cat(list(model.size_parameters())).data.cpu().numpy())
                optimizer.zero_grad()
                (criterion(output, labels) + lamb * model.capacity()).backward()
                optimizer.step()


if __name__ == '__main__':
    pass
