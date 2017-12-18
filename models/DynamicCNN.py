import torch
from torch import nn
from torch.nn.init import kaiming_uniform, normal
from modules.dynamic import Linear, Conv2d, Flatten

class DynamicCNN(nn.Module):
 
    def __init__(self, conv_layers, in_features=(1, 28,28), out_features=10, pool_size=2, conv_kernel=3, conv_initial_size=50, fc_initial_size=500, activation=nn.ReLU, pad=0, stride=1, device_id=0):
        super(DynamicCNN, self).__init__()
        layers = []
        self.in_features=in_features
        if pad > 0:
            self.pad = nn.ConstantPad2d(pad, 0)
        else:
            self.pad = None
        if pool_size > 0:
            pool = nn.MaxPool2d(pool_size)
        else:
            pool = None

        for i in range(conv_layers):
            layer = Conv2d(conv_kernel, stride=stride, weight_initializer=kaiming_uniform, bias_initializer=normal)
            layer.initial_size = conv_initial_size
            if in_features is not None:
                layer.in_features = in_features[0]
                layer.additional_dims = in_features[1:]
            layer.set_device_id(device_id)
            layers.append(layer)
            layers.append(activation())
            if pool is not None:
                layers.append(pool)
            in_features = None
        layers.append(Flatten())
        prefinal = Linear(weight_initializer=kaiming_uniform)
        prefinal.set_device_id(device_id)
        prefinal.initial_size=fc_initial_size
        layers.append(prefinal)
        layers.append(activation())
        last = Linear(out_features=out_features, weight_initializer=kaiming_uniform, bias_initializer=normal)
        last.set_device_id(device_id)
        layers.append(last)
        self.processor = nn.Sequential(*layers)

    def forward(self, x=None):
        if x is not None:
            x = x.view(*((-1,) + self.in_features))
        else:
            x = self.processor[0].generate_input()
        if self.pad is not None:
            x = self.pad(x)
        return self.processor(x)
