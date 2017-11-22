import torch
from torch import nn
from torch.nn.init import kaiming_uniform, normal
from modules.dynamic import Linear, DynamicModule

class MultiLayerDynamicPerceptron(nn.Module):
 
    def __init__(self, hidden_layers, in_features=28*28, out_features=10, initial_size = 10, activation=nn.ReLU, device_id=0):
        super(MultiLayerDynamicPerceptron, self).__init__()
        self.initial_size = initial_size
        layers = []
        self.in_features = in_features
        for _ in range(hidden_layers):
            layer = Linear(in_features=in_features, weight_initializer=kaiming_uniform, bias_initializer=normal)
            layer.set_device_id(device_id)
            layers.append(layer)
            layers.append(activation())
            in_features = None
        last = Linear(in_features=in_features, out_features=out_features, weight_initializer=kaiming_uniform, bias_initializer=normal)
        last.set_device_id(device_id)
        layers.append(last)
        self.processor = nn.Sequential(*layers)

    def forward(self, x=None):
        if x is not None:
            x = x.view(-1, self.in_features)
        return self.processor(x)
