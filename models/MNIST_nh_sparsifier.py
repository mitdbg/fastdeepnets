import torch
from torch import nn
from modules.LinearSparsifier import LinearSparsifier

from utils.misc import tn


class MNIST_nh_sparsifier(nn.Module):
 
    def __init__(self, hidden_layers, max_width):
        super(MNIST_nh_sparsifier, self).__init__()
        layers = []
        cin = 28*28
        for _ in range(hidden_layers):
            layers.append(LinearSparsifier(cin, max_width))
            layers.append(nn.ReLU())
            cin = max_width
        layers.append(nn.Linear(cin, 10))
        self.processor = nn.Sequential(*layers)

    def forward(self, x):
        return self.processor(x.view(-1, 28 * 28))

    def loss(self):
        losses = []
        for model in self.modules():
            if isinstance(model, LinearSparsifier):
                losses.append(model.loss())
        return sum(losses[1:], losses[0])

    def get_capacities(self):
        caps = []
        for model in self.modules():
            if isinstance(model, LinearSparsifier):
                caps.append(model.get_capacity())
        return torch.cat(caps)

    def get_capacity(self):
        return self.get_capacities().sum()

    def has_collapsed(self):
        return tn(self.get_capacities().min().data) == 0
