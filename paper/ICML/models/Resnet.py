import torch.nn as nn
from torch.nn import ReLU
import math
from dynnet.layers import (
    Input, Linear, BatchNorm, Flatten, CapNorm, Conv2d, Sum)
from dynnet.graph import Graph
from dynnet.filters import SmoothFilter

RESNET_CFGS = {
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3]
}

EXPANSION = 4

def block(graph, inp, planes, downsample, stride=1):
    conv1 = graph.add(Conv2d, out_channels=planes, kernel_size=1,
                      bias=False)(inp)
    bn1 = graph.add(BatchNorm)(conv1)
    filt1 = graph.add(SmoothFilter)(bn1)
    conv2 = graph.add(Conv2d, out_channels=planes, kernel_size=3,
                      stride=stride, padding=1, bias=False)(filt1)
    bn2 = graph.add(BatchNorm)(conv2)
    filt2 = graph.add(SmoothFilter)(bn2)
    conv3 = graph.add(Conv2d, out_channels=planes * EXPANSION,
                      kernel_size=1, bias=False)(filt2)
    bn3 = graph.add(BatchNorm)(conv3)
    ss = graph.add(Sum)(bn3, downsample)
    return graph.add(nn.ReLU, inplace=True)(ss)


class ResNet(nn.Module):

    def __init__(self, params):
        super(ResNet, self).__init__()
        num_classes = params['num_classes']
        layers = RESNET_CFGS[params['config']]
        factor = params['factor']
        self.graph = Graph()
        self.inplanes = 64 * factor
        self.inp = self.graph.add(Input, 3, 224, 224)()
        conv1 = self.graph.add(Conv2d, out_channels=self.inplanes,
                               kernel_size=7, stride=2, padding=3,
                               bias=False)(self.inp)
        filt1 = self.graph.add(SmoothFilter)(conv1)
        bn1 = self.graph.add(BatchNorm)(filt1)
        relu = self.graph.add(ReLU, inplace=True)(bn1)
        pool = self.graph.add(nn.MaxPool2d, kernel_size=3,
                              stride=2, padding=1)(relu)
        layer1 = self._make_layer(pool, 64 * factor, layers[0])
        layer2 = self._make_layer(layer1, 128 * factor, layers[1], stride=2)
        layer3 = self._make_layer(layer2, 256 * factor, layers[2], stride=2)
        layer4 = self._make_layer(layer3, 512 * factor, layers[3], stride=2)
        avgpool = self.graph.add(nn.AvgPool2d, 7, stride=1)(layer4)
        flat = self.graph.add(Flatten)(avgpool)
        self.result = self.graph.add(Linear, out_features=num_classes)(flat)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, inp, planes, blocks, stride=1):
        downsample = inp
        if stride != 1 or self.inplanes != planes * EXPANSION:
            downsample = self.graph.add(Conv2d, 
                                        out_channels=planes * EXPANSION,
                                        kernel_size=1, stride=stride,
                                        bias=False)(inp)
            downsample = self.graph.add(BatchNorm)(downsample)

        inp = block(self.graph, inp, planes, downsample, stride)
        self.inplanes = planes * EXPANSION
        for i in range(1, blocks):
            inp = block(self.graph, inp, planes, downsample)
        return inp

    def forward(self, x):
        return self.graph({self.inp: x}, self.result)[0]
