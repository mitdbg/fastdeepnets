"""This module implements different algorithms for feature selection"""

from typing import Any, Union
from torch import (
    ones, nonzero, LongTensor, ByteTensor, rand, randn, zeros
)
from torch.nn import Parameter
from torch.nn.functional import relu

from dynnet.interfaces import DynamicModule, GarbageCollectionLog
from dynnet.operations import IndexSelectOperation

class Filter(DynamicModule):

    def get_weights(self):
        raise NotImplementedError("This is an abstract class")


class SimpleFilter(Filter):

    def __init__(self, starting_value: Union[float, str] = 'random', **kwargs):
        input_features = kwargs['input_features']
        assert len(input_features) == 1, "Filters need 1 and 1 parent"
        features = input_features[0]
        # We are not changing the feature set
        kwargs['output_features'] = features
        super(SimpleFilter, self).__init__(**kwargs)
        if starting_value == 'random':
            # We do 1 - rand because, rand outputs in the range [0, 1) and
            # we do not want dead neurons from the beginning so (0, 1] is
            # a better range for the random numbers
            self.weight = Parameter(1 - rand(features.feature_count))
        elif starting_value == 'normal':
            self.weight = Parameter(randn(features.feature_count))
        elif starting_value == 'uniform':
            self.weight = Parameter(rand(features.feature_count) * 2 - 1)
        else:
            print(starting_value)
            self.weight = Parameter(ones(features.feature_count) * starting_value)

    def get_weights(self):
        return relu(self.weight)

    def forward(self, x):
        # Size checks
        super(SimpleFilter, self).forward(x)
        # We are doing this weird expand-based implementation because
        # This is the only implementation that keeps the output
        # contiguous in memory (and therefore does not involve reordering
        # it later
        weight = self.get_weights().unsqueeze(0)
        for _ in range(len(self.output_features.additional_dims)):
            weight = weight.unsqueeze(2)
        weight = weight.expand(x.size())
        x = x * weight
        return x

    def remove_input_features(self, remaining_features: LongTensor,
                              input_index: Any,
                              log: GarbageCollectionLog) -> None:
        assert input_index == 0, "We are only aware of one parent"
        # Let's reuse the logic
        self.remove_output_features(remaining_features, log)

    def remove_output_features(self, remaining_features: LongTensor,
                               log: GarbageCollectionLog) -> None:
        assert remaining_features.size(0) < self.weight.size(0), (
            "We should be removing features")
        operation = IndexSelectOperation(remaining_features, 0)
        self.weight = log.change_parameter(self.weight, operation)

    def get_alive_features(self) -> ByteTensor:
        """Mask containing ones when alive

        Returns
        -------
        The binary mask
        """
        return (self.weight.data > 0).cpu().squeeze()

    def garbage_collect(self, log: GarbageCollectionLog):
        non_zero_features = nonzero(self.get_alive_features()).squeeze()
        print(non_zero_features.size())
        self.output_features.remove_features(self, non_zero_features, log)
        # The input and the output feature bag is the same object
        # There is no need to update it

    def __repr__(self):
        return "SimpleFilter(%s)" % self.output_features.feature_count

class SmoothFilter(SimpleFilter):

    def __init__(self, starting_value: Union[float, str] = 'random',
                 gamma=0.99, threshold=0.5, **kwargs):
        self.gamma = gamma
        self.threshold = threshold
        super(SmoothFilter, self).__init__(starting_value, **kwargs)
        self.register_buffer('exp_avg', self.weight.data.sign().float())
        self.register_buffer('exp_std', zeros(self.weight.size()))
        self.register_buffer('mask', ByteTensor(self.weight.size()))
        self.mask.fill_(1)

    def get_weights(self):
        return self.weight

    def get_alive_features(self) -> ByteTensor:
        """Mask containing ones when alive

        Returns
        -------
        The binary mask
        """
        return self.mask

    def remove_input_features(self, remaining_features: LongTensor,
                              input_index: Any,
                              log: GarbageCollectionLog) -> None:
        assert input_index == 0, "We are only aware of one parent"
        # Let's reuse the logic
        self.remove_output_features(remaining_features, log)

    def remove_output_features(self, remaining_features: LongTensor,
                               log: GarbageCollectionLog) -> None:
        assert remaining_features.size(0) < self.weight.size(0), (
            "We should be removing features")
        operation = IndexSelectOperation(remaining_features, 0)
        self.weight = log.change_parameter(self.weight, operation)
        for buffer_name in ['exp_std', 'exp_avg', 'mask']:
            setattr(self, buffer_name, operation(getattr(self, buffer_name)))

    def update_statistics(self):
        gamma = self.gamma
        bs = self.get_weights().data.sign()
        diff = bs - self.exp_avg
        self.exp_std.mul_(gamma).addcmul_(1 - gamma, diff, diff)
        self.exp_avg.mul_(gamma).add_(1 - gamma, bs)
        self.mask.mul_(self.exp_std <= self.threshold)
        self.weight.data.mul_(self.mask.float())

    def __repr__(self):
        return "SmoothFilter(%s, gamma=%s)" % (
            self.output_features.feature_count,
            self.gamma
        )

