from dynnet.layers import Linear, Input, BatchNorm1d, Conv2d
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

# Fake input
x = Variable(torch.rand(3, 15))
xx = Variable(torch.rand(3, 3, 16, 16))

simple = Graph()
i = simple.add(Input, 3, 16, 16)()
o = simple.add(MaxPool2d, kernel_size=2)(i)



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
gc_log.update_optimizer(optim)

forward(graph, {inp: x}, fc2, optim)
print(graph)

graph2 = Graph()
inp2 = graph2.add(Input, 3, 5, 5)()
conv1 = graph2.add(Conv2d, out_channels=20, kernel_size=3)(inp2)
x2 = Variable(torch.rand(2, 3, 5, 5))
print(graph2)
result, = graph2({inp2: x2}, conv1)

