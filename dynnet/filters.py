"""This module implements different algorithms for feature selection"""

from typing import Any, Union
from torch import ones, nonzero, LongTensor, ByteTensor, rand
from torch.nn import Parameter
from torch.nn.functional import relu

from dynnet.interfaces import DynamicModule, GarbageCollectionLog
from dynnet.operations import IndexSelectOperation


class SimpleFilter(DynamicModule):

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
        else:
            self.weight = Parameter(ones(features.feature_count) * starting_value)

    def forward(self, x):
        # Size checks
        super(SimpleFilter, self).forward(x)
        # We are doing this weird expand-based implementation because
        # This is the only implementation that keeps the output
        # contiguous in memory (and therefore does not involve reordering
        # it later
        weight = self.weight.unsqueeze(0)
        for _ in range(len(self.output_features.additional_dims)):
            weight = weight.unsqueeze(2)
        weight = weight.expand(x.size())
        x = x * relu(weight)
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
