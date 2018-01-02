"""This module implements different algorithms for feature selection"""

from typing import Any
from torch import ones, nonzero, LongTensor, ByteTensor
from torch.nn import Parameter
from torch.nn.functional import relu

from dynnet.interfaces import DynamicModule, GarbageCollectionLog
from dynnet.operations import IndexSelectOperation


class SimpleFilter(DynamicModule):

    def __init__(self, starting_value: float = 1, **kwargs):
        input_features = kwargs['input_features']
        assert len(input_features) == 1, "Filters need 1 and 1 parent"
        features = input_features[0]
        # We are not changing the feature set
        kwargs['output_features'] = features
        super(SimpleFilter, self).__init__(**kwargs)
        self.weight = Parameter(ones(features.feature_count) * starting_value)

    def forward(self, x):
        # Size checks
        super(SimpleFilter, self).forward(x)
        dims = len(x.size())
        x = x.transpose(1, dims - 1)
        x = x * relu(self.weight)
        x = x.transpose(1, dims - 1)
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
        self.output_features.remove_features(self, non_zero_features, log)
        # The input and the output feature bag is the same object
        # There is no need to update it

    def __repr__(self):
        return "SimpleFilter(%s)" % self.output_features.feature_count
