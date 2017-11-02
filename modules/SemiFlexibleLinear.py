import torch
from torch.autograd import Variable
from torch import nn
from torch.nn.functional import sigmoid

def stable_logexp(factor, k):
    f = factor * k
    if f.data.cpu().numpy()[0] < 0:
        return (f.exp() + 1).log() / k
    else:
        return (1 + (-f).exp()).log() / k + factor

def compute_loss(k, x_0, size):
        return stable_logexp(x_0, k) - stable_logexp(x_0 - size, k)

class SemiFlexibleLinear(nn.Linear):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`

    The difference with a normal Linear layer is that a portion of the output
    will be set to zero depending on the learnable `size` parameter

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: True
        original_size: the starting effective size of the layer.

    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            (out_features x in_features)
        bias:   the learnable bias of the module of shape (out_features)
        size: The current effective size of the netowrk. All components of the
            output bigger than `size` will be set to zero (smoothly)
        range: the range from 0 to `out_features` to avoid regenerating it
            at every iteration
    """
    def __init__(self, in_features, out_features, bias=True, original_size=None):
        if original_size is None:
            original_size = out_features
        super(SemiFlexibleLinear, self).__init__(in_features, out_features, bias)
        self.size = nn.Parameter(torch.ones(1) * original_size)
        self.range = Variable(torch.arange(0, out_features), requires_grad=False)

    def cuda(self, device_id=None):
        result = super(SemiFlexibleLinear, self).cuda(device_id)
        result.range = self.range.cuda(device_id)
        return result

    def get_scaler(self):
        r"""Return the vector by which the output is multiplied

        It starts by ones, then there is the gray zones with values between
            0 and 1 and then zeros until the end of the vector
        """
        return sigmoid(self.size - self.range)

    def forward(self, inp):
        inp = super(SemiFlexibleLinear, self).forward(inp)
        inp = inp * self.get_scaler()
        return inp

    def capacity(self):
        r"""Return the current capacity used by the network

        It is a smooth number of neuros
        It is between 0 and `out_features`

        We suggest that the global loss should try to minimize the capacity
        of this layer
        """
        return compute_loss(1, self.size, self.out_features)
