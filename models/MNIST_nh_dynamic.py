import torch
from torch import nn
from torch.nn.init import kaiming_uniform, normal
from modules.dynamic import Linear, DynamicModule



class MNIST_nh_dynamic(nn.Module):
 
    def __init__(self, hidden_layers, initial_size = 10, device_id=0):
        super(MNIST_nh_dynamic, self).__init__()
        layers = []
        in_features = 28 * 28
        for _ in range(hidden_layers):
            layer = Linear(in_features=in_features, weight_initializer=kaiming_uniform, bias_initializer=normal)
            layer.set_device_id(0)
            layer.grow(initial_size)
            layers.append(layer)
            layers.append(nn.ReLU())
            in_features = None
        last = Linear(in_features=in_features, out_features=10, weight_initializer=kaiming_uniform, bias_initializer=normal)
        last.set_device_id(0)
        last.grow()
        layers.append(last)
        self.processor = nn.Sequential(*layers)

    def forward(self, x=None):
        if x is not None:
            x = x.view(-1, 28*28)
        return self.processor(x)
