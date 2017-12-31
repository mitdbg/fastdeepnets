"""This package contains simple neural network layers"""
from typing import Any
from torch import LongTensor
from torch.nn import (
    Linear as SimpleLinear,
    BatchNorm1d as SimpleBatchNorm1d
)

from dynnet.interfaces import DynamicModule, GarbageCollectionLog, FeatureBag
from dynnet.operations import IndexSelectOperation


class Input(DynamicModule):
    """Represent a neural network input

    Its sole use is to provide feature ids to layers down the graph
    """

    def __init__(self, input_count, graph=None, input_features=None):
        assert not input_features, "Input layer should have no parent"
        feature_bag = FeatureBag(input_count)

        super(Input, self).__init__(input_features=input_features,
                                    output_features=feature_bag,
                                    graph=graph)

    def forward(self, value):
        """This layer does no do anything"""
        return value  # Just forwarding the values

    def garbage_collect(self, log: GarbageCollectionLog):
        pass  # This layer never remove features

    def __repr__(self):
        return "Input(%s)" % self.output_features.feature_count


class BatchNorm1d(DynamicModule):
    """Applies Batch Normalization over a 2d or 3d input that is seen as a
    mini-batch.

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Args:
        num_features: num_features from an expected input of size
            `batch_size x num_features [x width]`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to true, gives the layer
            learnable affine parameters.

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm1d(100, affine=False)
        >>> input = autograd.Variable(torch.randn(20, 100))
        >>> output = m(input)
    """

    def __init__(self, *args, **kwargs):
        graph = kwargs['graph']
        input_features = kwargs['input_features']
        assert len(input_features) == 1, (
            "BatchNorm1d only supports 1 parent")
        del kwargs['graph']
        del kwargs['input_features']
        super(BatchNorm1d, self).__init__(graph=graph,
                                          input_features=input_features,
                                          output_features=input_features[0])
        self.implementation = SimpleBatchNorm1d(
            num_features=input_features[0].feature_count,
            *args, **kwargs)

    def garbage_collect(self, log: GarbageCollectionLog):
        pass  # This layer never remove features

    def remove_input_features(self, remaining_features: LongTensor,
                              input_index: Any,
                              log: GarbageCollectionLog) -> None:
        assert input_index == 0, "We are only aware of one parent"
        # Reuse logic (the input and the output features are the
        # same
        self.remove_output_features(remaining_features, log)

    def remove_output_features(self, remaining_features: LongTensor,
                               log: GarbageCollectionLog) -> None:
        operation = IndexSelectOperation(remaining_features, 0)
        self.implementation.weight = log.change_parameter(
            self.implementation.weight, operation)
        if self.implementation.bias is not None:
            self.implementation.bias = log.change_parameter(
                self.implementation.bias, operation)
        # The operations on the buffer do not need te be logged, they
        # are purely internals to the module
        self.implementation.running_mean = operation(
            self.implementation.running_mean)
        self.implementation.running_var = operation(
            self.implementation.running_var)
        self.implementation.num_features = (
            self.output_features.feature_count)

    def forward(self, *args):
        # Basic forwarding to the actual implementation
        return self.implementation(*args)

    def __repr__(self):
        return "Dyn%s" % self.implementation

class NaiveWrapper(DynamicModule):
    """This class wraps classic Pytorch modules into dynamic ones

    WARNING: These modules needs to have no state otherwise you will
    run into dimension problems after garbage collection. Layers that
    have state that depends on the size of the inputs needs to be
    properly implemented (especially their garbage collection routine)

    These layers also only support a single parent
    """

    def __init__(self, factory, *args, **kwargs):
        graph = kwargs['graph']
        input_features = kwargs['input_features']
        assert len(input_features) == 1, (
            "NaiveWrapper only supports 1 parent")
        del kwargs['graph']
        del kwargs['input_features']
        super(NaiveWrapper, self).__init__(graph=graph,
                                           input_features=input_features,
                                           output_features=input_features[0])
        self.implementation = factory(*args, **kwargs)

    def garbage_collect(self, log: GarbageCollectionLog):
        pass  # This layer never remove features

    def remove_input_features(self, remaining_features: LongTensor,
                              input_index: Any,
                              log: GarbageCollectionLog) -> None:
        assert input_index == 0, "We are only aware of one parent"
        # This layer should not have state, therefore cannot be updated
        pass

    def remove_output_features(self, remaining_features: LongTensor,
                               log: GarbageCollectionLog) -> None:
        assert False, "It is impossible to remove features from an Input layer"

    def forward(self, *args):
        # Basic forwarding to the actual implementation
        return self.implementation(*args)

    def __repr__(self):
        return "Dyn%s" % self.implementation


class Linear(DynamicModule):
    """Applies a linear transformation to the incoming data: :math:`y = Ax + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: True

    Shape:
        - Input: :math:`(N, in\_features)`
        - Output: :math:`(N, out\_features)`

    Attributes:
        weight: the learnable weights of the module of shape
            (out_features x in_features)
        bias:   the learnable bias of the module of shape (out_features)

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = autograd.Variable(torch.randn(128, 20))
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, *args, **kwargs):
        graph = kwargs['graph']
        input_features = kwargs['input_features']
        assert len(input_features) == 1, "Linear accepts 1&1 parent"
        # Use need to choose the number of defaut starting features for
        # each fully connected layer
        assert 'out_features' in kwargs, (
            "For Linear layers, out_features needs to be defined")
        out_features = kwargs['out_features']
        output_features = FeatureBag(out_features)
        super(Linear, self).__init__(graph=graph,
                                     output_features=output_features,
                                     input_features=input_features)
        # Keeping only the relevant arguments for the real implementation
        del kwargs['graph']
        del kwargs['input_features']
        # Pass the known arguments to the constructor
        kwargs['in_features'] = input_features[0].feature_count

        # Instantiating a Linear layer
        # Using composition instead of inheritance
        self.implementation = SimpleLinear(*args, **kwargs)

    def forward(self, x):
        return self.implementation(x)

    def remove_input_features(self, remaining_features: LongTensor,
                              input_index: Any,
                              log: GarbageCollectionLog) -> None:
        assert input_index == 0, "We are only aware of one parent"
        current_input_size = self.implementation.weight.size(1)
        assert current_input_size > remaining_features.size(0), (
            "We should be removing features")
        operation = IndexSelectOperation(remaining_features, 1)
        self.implementation.weight = (
            log.change_parameter(self.implementation.weight, operation))
        self.implementation.in_features = remaining_features.size(0)

    def remove_output_features(self, remaining_features: LongTensor,
                               log: GarbageCollectionLog) -> None:
        current_output_size = self.implementation.weight.size(0)
        assert current_output_size > remaining_features.size(0), (
            "We should be removing features")
        operation = IndexSelectOperation(remaining_features, 0)
        self.implementation.weight = (
            log.change_parameter(self.implementation.weight, operation))
        # Only update the bias if this layer uses one
        if self.implementation.bias is not None:
            self.implementation.bias = (
                log.change_parameter(self.implementation.bias, operation))
        self.implementation.out_features = remaining_features.size(0)

    def garbage_collect(self, log: GarbageCollectionLog):
        pass  # This layer never remove features

    def __repr__(self):
        return "Dyn%s" % self.implementation
