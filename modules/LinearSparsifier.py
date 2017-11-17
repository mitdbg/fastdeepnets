import torch
from torch import nn


class LinearSparsifier(nn.Linear):
    r"""Applies a linear transformation to the incoming data: :math:`y = factor * (Ax + b)`

    The difference with a normal Linear layer is that a portion of the output
    will be set to zero depending on the learnable `filter` parameter

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: True

    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            (out_features x in_features)
        bias:   the learnable bias of the module of shape (out_features)
        filter: the vector representing the liveness of each neurons
    """
    def __init__(self, in_features, out_features, bias=True):
        super(LinearSparsifier, self).__init__(in_features, out_features, bias)
        self.filter = nn.Parameter(torch.ones(out_features))
        self.filter_activation = nn.ReLU()

    def get_filter(self):
        r""" Return the actual factor to apply on each neuron.

        It is necessarly a positive value
        """
        return self.filter_activation(self.filter)

    def forward(self, inp):
        inp = super(LinearSparsifier, self).forward(inp)
        inp = inp * self.get_filter()
        return inp

    def loss(self):
        r"""Return the loss you want to apply to make this layer sparse
        """
        return self.get_filter().sum()

    def get_alive_neurons(self):
        r"""Return a binary mask of each alive neuron
        """
        return self.get_filter() > 0

    def get_capacity(self):
        r"""Return the number of neurons this layer effectively uses
        """
        return self.get_alive_neurons().float().sum()
