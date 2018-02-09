import torch
from dynnet.graph import Sequential
from dynnet.layers import Input, Linear, BatchNorm, Flatten, CapNorm
from dynnet.filters import SmoothFilter

class FullyConnected(torch.nn.Module):

    def __init__(self, params):
        super(FullyConnected, self).__init__()
        layer_count = params['layers']
        dropout = params['dropout']
        batch_norm = params['batch_norm']
        dynamic = params['dynamic']
        input_features = params['input_features']
        output_features = params['output_features']
        gamma = params['gamma']
        graph = Sequential()
        graph.add(Input, *input_features)
        if len(input_features) > 1:
            graph.add(Flatten)

        Bn = BatchNorm

        assert layer_count > 0, "Need at least one layer"
        for i in range(layer_count):
            graph.add(Linear, out_features=params['size_layer_%s' % (i + 1)])
            if batch_norm:
                graph.add(Bn)
            if dropout > 0:
                graph.add(torch.nn.Dropout, p=dropout)
            if dynamic:
                graph.add(SmoothFilter, starting_value='uniform', gamma=gamma)
            graph.add(torch.nn.ReLU, inplace=True)
        graph.add(Linear, out_features=output_features)
        self.graph = graph

    def forward(self, x):
        return self.graph(x)

    def garbage_collect(self):
        return self.graph.garbage_collect()
