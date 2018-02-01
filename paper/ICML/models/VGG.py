import torch
import torch.nn as nn
from torch.nn import ReLU
from torch.autograd import Variable
from dynnet.layers import Input, Linear, BatchNorm, Flatten, CapNorm, Conv2d
from dynnet.filters import SmoothFilter
from dynnet.graph import Sequential

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, params):
        dynamic = params['dynamic']
        batch_norm = params['batch_norm']
        super(VGG, self).__init__()
        if dynamic:
            Bn = BatchNorm
        else:
            Bn = BatchNorm

        graph = Sequential()
        self.graph = graph
        graph.add(Input, *params['input_features'])

        config = cfg[params['name']]

        for descriptor in config:
            if descriptor == 'M':
                graph.add(nn.MaxPool2d, kernel_size=2, stride=2)
            else:
                descriptor = int(descriptor * params['factor'])
                graph.add(Conv2d, out_channels=descriptor,
                          kernel_size=3, padding=1)
                if batch_norm:
                    graph.add(Bn)
                if dynamic:
                    graph.add(SmoothFilter, starting_value='uniform',
                              gamma=params['gamma'])
                graph.add(ReLU, inplace=True)
        graph.add(Flatten)
        for i in range(2):
            graph.add(Linear, out_features=params['classifier_layer_%s' % (i + 1)])
            graph.add(SmoothFilter, starting_value='uniform',
                      gamma=params['gamma'])
            graph.add(ReLU, inplace=True)
        graph.add(Linear, out_features=params['output_features'])

    def forward(self, x):
        return self.graph(x)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def garbage_collect(self):
        return self.graph.garbage_collect()
