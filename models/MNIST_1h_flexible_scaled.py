import torch
from torch.autograd import Variable
from torch import nn
from torch.nn.functional import sigmoid

class MNIST_1h_flexible_scaled(nn.Module):
    def __init__(self, size, wrap, x=None):
        super(MNIST_1h_flexible_scaled, self).__init__()
        if x is None:
            x = size
        self.size = size
        self.x_0 = nn.Parameter(torch.ones(1) * x)
        self.scaler = nn.Parameter(torch.ones(size))
        self.hidden_layer = nn.Linear(28 * 28, size, bias=True)
        self.activation = nn.ReLU()
        self.range = wrap(Variable(torch.arange(0, size), requires_grad=False))
        self.output_layer = nn.Linear(size, 10, bias=True)

    def get_selector(self):
        return sigmoid(-(self.range - self.x_0))

    def get_total_factor(self):
        return self.get_selector() * self.scaler

    def partial_forward(self, x):
        x = x.view(-1, 28*28)
        x = self.hidden_layer(x)
        x = self.activation(x)
        return x

    def forward(self, x):
        x = self.partial_forward(x)
        x = x * self.get_total_factor()
        x = self.output_layer(x)
        return x

    def loss(self):
        return self.x_0

    def reorder(self):
        total_weights = self.get_total_factor()
        self.range.data.set_(total_weights.sort(0, descending=True)[1].data.float())
        self.scaler.data.set_((total_weights / self.get_selector()).data)
        self.scaler.data[(self.scaler != self.scaler).data] = 1
        self.scaler.data.clamp_(min=1e-1, max=1e1)
