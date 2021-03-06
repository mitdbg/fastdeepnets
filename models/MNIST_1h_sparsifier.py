import torch
from torch import nn

class MNIST_1h_sparsifier(nn.Module):
    def __init__(self, max_size, strict=True):
        super(MNIST_1h_sparsifier, self).__init__()
        self.strict = strict
        self.max_size = max_size
        self.hidden_layer = nn.Linear(28*28, max_size, bias=True)
        self.activation = nn.ReLU()
        self.filter = nn.Parameter(torch.ones(max_size))
        self.filter_activation = nn.ReLU()
        self.output_layer = nn.Linear(max_size, 10, bias=True)
        self.stats_count = 0

    def get_filter(self):
        return self.filter_activation(self.filter)

    def partial_forward(self, x):
        x = self.hidden_layer(x.view(-1, 28*28))
        x = self.activation(x)
        x = x * self.get_filter()
        return x

    def forward(self, x):
        x = self.partial_forward(x)
        x = self.output_layer(x)
        return x

    def loss(self):
        if self.strict:
            return self.get_filter().sum()
        else:
            return torch.abs(self.filter).sum()

    def get_alive_neurons(self):
        return self.get_filter() > 0

    def l0_loss(self):
        return self.get_alive_neurons().float().sum()
