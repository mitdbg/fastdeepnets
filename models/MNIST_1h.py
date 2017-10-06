from torch import nn


class MNIST_1h(nn.Module):
    def __init__(self, size):
        super(MNIST_1h, self).__init__()
        self.size = size
        self.hidden_layer = nn.Linear(28 * 28, size, bias=True)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(size, 10, bias=True)

    def forward(self, x):
        return self.output_layer(self.partial_forward(x))

    def partial_forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.activation(self.hidden_layer(x))

