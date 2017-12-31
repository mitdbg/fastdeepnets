from dynnet.layers import Linear, Input, BatchNorm1d
from dynnet.filters import SimpleFilter
from dynnet.graph import Graph
import torch
from torch.nn import Dropout
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
