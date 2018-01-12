from dynnet.layers import Linear, Input, BatchNorm1d, Conv2d, Flatten
from dynnet.filters import SimpleFilter
from dynnet.graph import Sequential
import torch
from torch.nn import Dropout, MaxPool2d
from torch.autograd import Variable
from torch.optim import Adam
from time import time

def loss(tensor):
    return tensor.sum()

def forward(graph, inp, optimizer):
    output = graph(inp)
    optimizer.zero_grad()
    loss(output).backward()
    optimizer.step()

def remove_half_neurons(graph):
    for m in graph:
        if isinstance(m, SimpleFilter):
            m.weight.data.uniform_(-1, 1)

# Fully connected example
x = Variable(torch.rand(60, 28*28))
graph = Sequential()
graph.add(Input, 28*28)
graph.add(Linear, out_features=10000)
graph.add(SimpleFilter)
graph.add(Linear, out_features=10000)
graph.add(SimpleFilter)
graph.add(Linear, out_features=10000)
graph.add(SimpleFilter)
graph.add(Linear, out_features=10000)
graph.add(SimpleFilter)
graph.add(Linear, out_features=10)
optim = Adam(graph.parameters())
a = time()
forward(graph, x, optim)
print("forward pass", (time() - a) / x.size(0) * 60000)
remove_half_neurons(graph)
b = time()
gc_log = graph.garbage_collect()
print("main_gc", time() - b)
c = time()
# You need to think to update the optimizer
# (at least optimizers that keep state in them)
gc_log.update_optimizer(optim)
print("opt gc", time() - c)
d = time()
forward(graph, x, optim)
print("pruned forward pass", (time() - d) / x.size(0) * 60000)
