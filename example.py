from dynnet.layers import Linear, Input, BatchNorm1d, Conv2d, Flatten
from dynnet.filters import SimpleFilter
from dynnet.graph import Graph
import torch
from torch.nn import Dropout, MaxPool2d
from torch.autograd import Variable
from torch.optim import Adam

def loss(tensor):
    return tensor.sum()

def forward(graph, mapping, target, optimizer):
    output, = graph(mapping, target)
    optimizer.zero_grad()
    loss(output).backward()
    optimizer.step()

# Fully connected example
x = Variable(torch.rand(3, 15))
graph = Graph()
inp = graph.add(Input, 15)()
fc1 = graph.add(Linear, out_features=20)(inp)
filt = graph.add(SimpleFilter)(fc1)
fc2 = graph.add(Linear, out_features=3)(filt)
filt2 = graph.add(SimpleFilter)(fc2)
bn = graph.add(BatchNorm1d)(filt2)
dr = graph.add(Dropout, 0.3)(bn)
fc2 = graph.add(Linear, out_features=9)(dr)
print(graph)
optim = Adam(graph.parameters())
forward(graph, {inp: x}, fc2, optim)
filt.weight.data[2:5].zero_()
filt2.weight.data[1:2].zero_()
gc_log = graph.garbage_collect()
# You need to think to update the optimizer
# (at least optimizers that keep state in them)
gc_log.update_optimizer(optim)
forward(graph, {inp: x}, fc2, optim)
print(graph)

# CNN example
graph2 = Graph()
inp2 = graph2.add(Input, 3, 16, 16)()
conv1 = graph2.add(Conv2d, out_channels=20, kernel_size=3)(inp2)
mp = graph2.add(MaxPool2d, kernel_size=2)(conv1)
conv_fil = graph2.add(SimpleFilter)(mp)
fl = graph2.add(Flatten)(conv_fil)
fc3 = graph2.add(Linear, out_features=10)(fl)
x2 = Variable(torch.rand(2, 3, 16, 16))
print(graph2)
result, = graph2({inp2: x2}, fc3)
conv_fil.weight.data[0:3].zero_()
graph2.garbage_collect()
print(graph2)


