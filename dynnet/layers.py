"""This package contains simple neural network layers"""
from typing import Any, Callable
import numpy as np
from torch import LongTensor, arange
from torch.autograd import Variable
from torch.nn import (
    Linear as SimpleLinear,
    BatchNorm1d as SimpleBatchNorm1d,
    Conv2d as SimpleConv2d
)

from dynnet.interfaces import (
    DynamicModule, GarbageCollectionLog, FeatureBag, MirrorFeatureBag
)
from dynnet.operations import IndexSelectOperation


class Input(DynamicModule):
    """Represent a neural network input

    Its sole use is to provide feature ids to layers down the graph
    """

    def __init__(self, *dimensions,
                 graph=None, input_features=None):
        """Create an Input layer

        Parameters
        ----------

        dimensions
            A list of dimensions for this input
        graph
            The computation graph it belongs to
        input_features
            The list of parent features
        """
        assert not input_features, "Input layer should have no parent"
        feature_bag = FeatureBag(*dimensions)

        super(Input, self).__init__(input_features=input_features,
                                    output_features=feature_bag,
                                    graph=graph)

    def forward(self, value):
        """This layer does no do anything except checking dimensions"""
        expected_dims = ((self.output_features.feature_count,) + (
            self.output_features.additional_dims))
        dimensions = value.size()[1:]
        assert expected_dims == dimensions, (
            "Invalid dimensions for Input layer, got %s, expected %s" % (
                dimensions, expected_dims))
        return value  # Just forwarding the values

    def garbage_collect(self, log: GarbageCollectionLog):
        pass  # This layer never remove features

    def __repr__(self):
        return "Input(%s)" % self.output_features.feature_count


class NaiveWrapper(DynamicModule):
    """This class wraps classic Pytorch modules into dynamic ones

    WARNING: These modules needs to have no state otherwise you will
    run into dimension problems after garbage collection. Layers that
    have state that depends on the size of the inputs needs to be
    properly implemented (especially their garbage collection routine)

    We infer the output size doing a forward pass, It might incur a small
    performance penalty on very complex layers

    These layers also only support a single parent
    """

    def __init__(self, factory, *args, **kwargs):
        graph = kwargs['graph']
        input_features = kwargs['input_features']
        assert len(input_features) == 1, (
            "NaiveWrapper only supports 1 parent")
        del kwargs['graph']
        del kwargs['input_features']
        implementation = factory(*args, **kwargs)
        sample_input = input_features[0].sample_typical_input()
        # We make it volatile because we won't be doing backprop on it
        sample_output = implementation(Variable(sample_input,
                                                volatile=True))
        # We discard the batch size and the meaningful feature
        sample_output_size = sample_output.size()[2:]
        output_features = MirrorFeatureBag(input_features[0],
                                           *sample_output_size)
        super(NaiveWrapper, self).__init__(graph=graph,
                                           input_features=input_features,
                                           output_features=output_features)
        self.implementation = implementation

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


class BaseDynamicLayer(DynamicModule):
    """This Layer is a common base for simple layer like Linear or
    Conv2d
    """

    def __init__(self, factory: Callable,
                 in_feature_arg_name: str,
                 out_feature_arg_name: str,
                 in_feature_dim: int,
                 out_feature_dim: int,
                 *args, **kwargs):
        self.in_feature_dim = in_feature_dim
        self.out_feature_dim = out_feature_dim
        self.in_feature_arg_name = in_feature_arg_name
        self.out_feature_arg_name = out_feature_arg_name
        graph = kwargs['graph']
        input_features = kwargs['input_features']
        assert len(input_features) == 1, "Simple Layers accept 1&1 parent"

        # Keeping only the relevant arguments for the real implementation
        del kwargs['graph']
        del kwargs['input_features']
        # Pass the known arguments to the constructor
        kwargs[in_feature_arg_name] = input_features[0].feature_count
        kwargs[out_feature_arg_name] = kwargs[out_feature_arg_name]

        implementation = factory(*args, **kwargs)

        input_features = input_features[0]
        sample_input = input_features.sample_typical_input()
        sample_output = implementation(Variable(sample_input,
                                                volatile=True))
        additional_dims = sample_output.size()[2:]
        # If the features are supposed to be the same we mirror the
        # feature bag
        if in_feature_dim == out_feature_dim:
            output_features = MirrorFeatureBag(input_features,
                                               *additional_dims)
        # If they are independant we instantiate a new feature bag
        else:
            output_features = FeatureBag(sample_output.size(1),
                                         *additional_dims)

        # Use need to choose the number of defaut starting features for
        # each fully connected layer
        super(BaseDynamicLayer, self).__init__(graph=graph,
                                               output_features=output_features,
                                               input_features=[input_features])

        # Instantiating a Linear layer
        # Using composition instead of inheritance
        self.implementation = implementation

    def garbage_collect(self, log: GarbageCollectionLog):
        pass  # This layer never remove features

    def __repr__(self):
        return "Dyn%s" % self.implementation

    def forward(self, x):
        return self.implementation(x)

    def remove_input_features(self, remaining_features: LongTensor,
                              input_index: Any,
                              log: GarbageCollectionLog) -> None:
        assert input_index == 0, "We are only aware of one parent"
        dim = self.in_feature_dim
        current_input_size = self.implementation.weight.size(dim)
        assert current_input_size > remaining_features.size(0), (
            "We should be removing features")
        operation = IndexSelectOperation(remaining_features, dim)
        self.implementation.weight = (
            log.change_parameter(self.implementation.weight, operation))
        setattr(self.implementation,
                self.in_feature_arg_name,
                remaining_features.size(0))

    def remove_output_features(self, remaining_features: LongTensor,
                               log: GarbageCollectionLog) -> None:
        dim = self.out_feature_dim
        current_output_size = self.implementation.weight.size(dim)
        assert current_output_size > remaining_features.size(0), (
            "We should be removing features")
        operation = IndexSelectOperation(remaining_features, dim)
        self.implementation.weight = (
            log.change_parameter(self.implementation.weight, operation))
        # Only update the bias if this layer uses one
        if self.implementation.bias is not None:
            self.implementation.bias = (
                log.change_parameter(self.implementation.bias, operation))
        setattr(self.implementation,
                self.out_feature_arg_name,
                remaining_features.size(0))


class Linear(BaseDynamicLayer):
    """We will reuse the docstring from pytorch"""

    def __init__(self, *args, **kwargs):
        # Use need to choose the number of defaut starting features for
        # each fully connected layer
        assert 'out_features' in kwargs, (
            "For Linear layers, out_features needs to be defined")
        assert len(kwargs['input_features']) == 1, (
            "Only one input allowed for Linear Layers"
        )
        super(Linear, self).__init__(factory=SimpleLinear,
                                     in_feature_arg_name="in_features",
                                     out_feature_arg_name="out_features",
                                     in_feature_dim=1,
                                     out_feature_dim=0,
                                     *args, **kwargs)


class BatchNorm1d(BaseDynamicLayer):
    """We will reuse the docstring from pytorch"""

    def __init__(self, *args, **kwargs):
        input_features = kwargs['input_features']
        assert len(input_features) == 1, (
            "BatchNorm1d only supports 1 parent")

        super(BatchNorm1d, self).__init__(factory=SimpleBatchNorm1d,
                                          in_feature_arg_name="num_features",
                                          out_feature_arg_name="num_features",
                                          in_feature_dim=0,
                                          out_feature_dim=0,
                                          *args, **kwargs)

    def remove_input_features(self, remaining_features: LongTensor,
                              input_index: Any,
                              log: GarbageCollectionLog) -> None:
        assert input_index == 0, "We are only aware of one parent"
        # Reuse logic (the input and the output features are the
        # same. We do not want to call super here because the base class
        # because otherwise we would not update the buffers, only the
        # parameters
        self.remove_output_features(remaining_features, log)

    def remove_output_features(self, remaining_features: LongTensor,
                               log: GarbageCollectionLog) -> None:
        super(BatchNorm1d, self).remove_output_features(remaining_features,
                                                        log)
        operation = IndexSelectOperation(remaining_features, 0)
        # The operations on the buffer do not need te be logged, they
        # are purely internals to the module
        self.implementation.running_mean = operation(
            self.implementation.running_mean)
        self.implementation.running_var = operation(
            self.implementation.running_var)
        self.implementation.num_features = (
            self.output_features.feature_count)


class Conv2d(BaseDynamicLayer):
    """We will reuse the docstring from pytorch"""

    def __init__(self, *args, **kwargs):
        # Use need to choose the number of defaut starting features for
        # each fully connected layer
        assert 'out_channels' in kwargs, (
            "For Conv layers, out_channels needs to be defined")
        super(Conv2d, self).__init__(factory=SimpleConv2d,
                                     in_feature_arg_name="in_channels",
                                     out_feature_arg_name="out_channels",
                                     in_feature_dim=1,
                                     out_feature_dim=0,
                                     *args, **kwargs)

class Flatten(DynamicModule):

    def __init__(self, *args, **kwargs):
        input_features = kwargs['input_features']
        assert len(input_features) == 1, (
            "The View layer only accepts one and exactly one parent")
        input_features = input_features[0]
        total_features = (input_features.feature_count *
                          np.array(input_features.additional_dims).prod())
        output_features = FeatureBag(total_features)
        graph = kwargs['graph']
        super(Flatten, self).__init__(input_features=[input_features],
                                      output_features=output_features,
                                      graph=graph)

    def forward(self, x):
        # Check input dimensions
        super(Flatten, self).forward(x)
        return x.view(x.size(0), -1)

    def remove_input_features(self, remaining_features: LongTensor,
                              input_index: Any,
                              log: GarbageCollectionLog) -> None:
        assert input_index == 0, "We are only aware of one parent"
        diff = int(np.array(self.input_features[0].additional_dims).prod())
        indicies = arange(0, diff).long().unsqueeze(1)
        indicies = indicies.repeat(1, remaining_features.size(0))
        indicies += remaining_features * diff
        indicies = indicies.view(-1)
        self.output_features.remove_features(self, indicies, log)

    def remove_output_features(self, remaining_features: LongTensor,
                               log: GarbageCollectionLog) -> None:
        # It is impossible to remove features bottom down. I wanted to
        # raise an exception here but the removal of input feature actually
        # removes output features. So you should not do it but we cannot always
        # catch bad user behavior. We can only do a very simple check: that
        # the output size is equivalent to the input size
        expected_features = self.input_features[0].feature_count
        expected_features *= (
            np.array(self.input_features[0].additional_dims).prod())
        assert remaining_features.size(0) == expected_features, (
            "You are not allowed to change the number of ouput size of a" +
            "Flatten layer, it means we would have to remove more than asked")
        # We are just checking that the request acutally does nothing
        # (We only want this function to be called as a result of a input
        # feature removal

    def __repr__(self):
        return "Flatten()"


# Fill documentation
BatchNorm1d.__doc__ = SimpleBatchNorm1d.__doc__
Linear.__doc__ = SimpleLinear.__doc__
Conv2d.__doc__ = SimpleConv2d.__doc__
