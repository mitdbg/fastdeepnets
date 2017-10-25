import torch
from torch.autograd import Variable
from torch import nn
from torch.nn.functional import sigmoid

def stable_logexp(factor, k):
    f = factor * k
    if f.data.cpu().numpy()[0] < 0:
        return (f.exp() + 1).log() / k
    else:
        return (1 + (-f).exp()).log() / k + factor

def compute_loss(k, x_0, size):
        return stable_logexp(x_0, k) - stable_logexp(x_0 - size, k)

class MNIST_1h_flexible(nn.Module):
    def __init__(self, size, wrap):
        super(MNIST_1h_flexible, self).__init__()
        self.size = size
        self.k = Variable(torch.ones(1), requires_grad=False)
        self.x_0 = nn.Parameter(torch.ones(1) * size)
        self.hidden_layer = nn.Linear(28 * 28, size, bias=True)
        self.activation = nn.ReLU()
        self.range = wrap(Variable(torch.arange(0, size), requires_grad=False))
        self.output_layer = nn.Linear(size, 10, bias=True)

    def get_scaler(self):
        return sigmoid(-self.k * (self.range - self.x_0))

    def partial_forward(self, x):
        x = x.view(-1, 28*28)
        x = self.hidden_layer(x)
        x = self.activation(x)
        return x

    def forward(self, x):
        x = self.partial_forward(x)
        x = x * self.get_scaler()
        x = self.output_layer(x)
        return x
    def loss(self):
        return compute_loss(self.k, self.x_0, self.size)
