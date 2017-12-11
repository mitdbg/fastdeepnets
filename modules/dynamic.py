import torch
import numpy as np
from torch import nn
from torch.nn.functional import linear, relu, conv2d
from torch.nn import Module, Parameter, ParameterList
from torch.autograd import Function, Variable
from torch.nn.modules.conv import _single, _pair, _triple
from utils.misc import tn
from torchvision.datasets import MNIST, FashionMNIST

def remove_parameter(optimizer, parameter):
    if optimizer is None:
        return
    for group in optimizer.param_groups:
        group['params'] = [x for x in group['params'] if x is not parameter]
    del optimizer.state[parameter]

def get_statistics(optimizer, parameter):
    if optimizer is None:
        return
    for stats in  optimizer.state[parameter].values():
        if torch.is_tensor(stats) and stats.size() == parameter.size():
            yield stats

def apply_patch(optimizer, parameter, patch, transpose_dims=None):
    if optimizer is None:
        return
    d = optimizer.state[parameter]
    for stat_key in list(d.keys()):
        value = d[stat_key]
        if torch.is_tensor(value) and value.size() == parameter.size():
            if transpose_dims is not None:
                new_value = value.transpose(*transpose_dims)[patch]
                new_value = new_value.transpose(*transpose_dims)
            else:
                new_value = value[patch]
            d[stat_key] = new_value

def update_reference(optimizer, old_ref, new_ref):
    if optimizer is None:
      return
    for group in optimizer.param_groups:
        group['params'] = [new_ref if x is old_ref else x for x in group['params']]
    stats = optimizer.state[old_ref]
    del optimizer.state[old_ref]
    optimizer.state[new_ref] = stats

def walk_graph_feature_ids(node):
    if hasattr(node, 'get_feature_ids'):
        return node.get_feature_ids()
    if hasattr(node, 'grad_fn'):
        return walk_graph_feature_ids(node.grad_fn)
    if hasattr(node, 'next_functions') and len(node.next_functions) == 1:
        return walk_graph_feature_ids(node.next_functions[0][0])
    return node

def compute_patch(old_features, new_features):
    old_features = old_features.numpy()
    new_features = new_features.numpy()
    selector = np.where(np.isin(old_features, new_features))
    if not np.all(old_features[selector] == new_features[np.isin(new_features, old_features)]):
        raise NotImplementedError('Feature Reorder is not supported')
    if len(selector[0]) == 0:
        return torch.zeros(0).long()
    return torch.from_numpy(selector[0])

class FeaturesIdProvider(Function):
    def __init__(self, feature_ids):
        self._feature_ids = feature_ids

    def forward(self,x):
        return x + 0 # Otherwise the function is optimized out...

    def backward(self, grad_output):
        return grad_output

    def get_feature_ids(self):
        return self._feature_ids

def default_initializer(tensor):
    return tensor.normal_()

class DynamicModule(Module):
    def __init__(self):
        super(DynamicModule, self).__init__()
        # State flags
        self.collecting = False
        self.growing = False
        self.max_feature = 0

        # Feature ids tracker
        self.in_features_map = []
        self.out_features_map = []
        self.additional_dims = ()

        # The optimizer to keep updated as we resize the network
        self.optimizer = None

    def grow(self, size=0):
        if self.growing is not False:
            # raise AssertionError('Already growing, do at least one pass')
            pass
        self.growing = size
        return self

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def garbage_collect(self):
        if self.collecting is not False:
            # raise AssertionError('Already collecting, do at least one pass')
            pass
        self.collecting = True
        return self

    def forward(self, x=None):
        if x is None:
            x = self.generate_input()
        # Check if any resize operation scheduled
        if self.collecting or self.growing is not False:
            if self._in_feature_ids is None:
                feature_ids = walk_graph_feature_ids(x)
            else:
                feature_ids = self._in_feature_ids

            if self.collecting:
                self.collect_now(feature_ids, x.size())
            if self.growing is not False:
                self.grow_now(feature_ids, x.size())

        x = self.compute(x)

        # Attach feature Ids to the graph
        x = FeaturesIdProvider(self._out_feature_ids)(x)

        return x


class WeightedDynamicModule(DynamicModule):

    def __init__(
        self,
        in_features=None,
        out_features=None,
        weight_allocation=(),
        weight_initializer=default_initializer,
        bias_initializer=default_initializer,
        reuse_features=True,
        bias=True
    ):
        super(WeightedDynamicModule, self).__init__()

        # Saving parameters
        self.in_features = in_features
        self.out_features = out_features
        self.weight_allocation = weight_allocation
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.reuse_features = reuse_features
        self.has_bias = bias

        if self.out_features is not None:
            # We have a fixed number of features out so we precompute it
            self._out_feature_ids = torch.arange(0, self.out_features).long()
        else:  # We will need to define it later
            self._out_feature_ids = None

        if self.in_features is not None:
            self._in_feature_ids = torch.arange(0, self.in_features).long()
        else:
            self._in_feature_ids = None

        # Weight Parameters
        self.weight_blocks = ParameterList()

        # Bias parameters
        if self.has_bias:
            self.bias_blocks = ParameterList()

        # Filter Parameter
        if self.out_features is None:
            self.filters_blocks = ParameterList()

        # Device information
        self.device_id = -1 # CPU

    def set_device_id(self, device_id):
        self.device_id = device_id

    def wrap(self, tensor):
        if self.device_id == -1:
            return tensor.cpu()
        else:
            return tensor.cuda(self.device_id)

    def regenerate_out_feature_ids(self):
        if self.out_features is None:
            non_empty_features = [x for x in self.out_features_map if len(x) > 0]
            if len(non_empty_features) == 0:
                self._out_feature_ids = self.wrap(torch.zeros(0).long())
            else:
                self._out_feature_ids = torch.cat(non_empty_features)

    def block_parameters(self, block_id=-1):  # By default the last block
        result = []
        if len(self.weight_blocks) == 0:
            return result
        result.append(self.weight_blocks[block_id])
        if hasattr(self,'bias_blocks'):
            result.append(self.bias_blocks[block_id])
        if hasattr(self, 'filters_blocks'):
            result.append(self.filters_blocks[block_id])
        return (x for x in result if len(x) > 0)  # remove dead parameters

    def grow_now(self, feature_ids, input_shape=None):
        # Compute the number of rows (outputs) in the weight matrix
        if self.out_features is None:
            rows = self.growing
            out_feature_ids = torch.arange(self.max_feature, self.max_feature + rows)
            self.out_features_map.append(out_feature_ids)
            self.max_feature += rows
        else:
            rows = self.out_features

        # Compute the number of columns (inputs) in the weight matrix
        if self.in_features is not None:
            cols = self.in_features
        else:
            if self.reuse_features or len(self.in_feature_maps) == 0:
                new_features = feature_ids.clone()
            else:
                m = self.in_feature_maps[-1].max()
                new_features = feature_ids[feature_ids > m]
            cols = len(new_features)
            self.in_features_map.append(new_features)

        weights = self.wrap(self.weight_initializer(
            torch.zeros((rows, cols) + self.weight_allocation)
        ))
        self.weight_blocks.append(Parameter(weights))

        if hasattr(self, 'bias_blocks'):
            bias = self.wrap(self.bias_initializer(torch.zeros(rows)))
            self.bias_blocks.append(Parameter(bias))

        if hasattr(self, 'filters_blocks'):
            filter = self.wrap(torch.ones(rows))
            self.filters_blocks.append(Parameter(filter))

        if self.out_features is None:
            new_feature_ids = []

        self.growing = False
        self.regenerate_out_feature_ids()

    def collect_now(self, feature_ids, input_shape=None):
        if not self.collecting:
            return

        # Collect old inputs
        if self.in_features is None: # Only process dynamic input models
            new_feature_ids = []
            new_weight_blocs = ParameterList()
            for old_features, weights in zip(self.in_features_map, self.weight_blocks):
                patch = compute_patch(old_features, feature_ids)
                empty = Parameter(torch.zeros(0))
                if len(patch) == 0: # dead block
                    new_feature_ids.append(empty.data.long())
                    new_weights = self.wrap(empty)
                    remove_parameter(self.optimizer, weights)
                else:
                    new_feature_ids.append(old_features[patch])
                    patch = self.wrap(patch)
                    last_dim = 1
                    new_weights = weights.data.transpose(0, last_dim)[patch].transpose(0, last_dim)
                    new_weights = Parameter(new_weights)
                    apply_patch(self.optimizer, weights, patch, (0, last_dim))
                    update_reference(self.optimizer, weights, new_weights)
                new_weight_blocs.append(new_weights)
            self.weight_blocks = new_weight_blocs
            self.in_features_map = new_feature_ids

        # Collect unused features
        if self.out_features is None: # Only process dynamicoutput models
            new_feature_ids = []
            new_weight_blocs = ParameterList()
            new_filters_blocks = ParameterList()
            new_bias_blocks = ParameterList()
            source = zip(self.out_features_map,
                self.weight_blocks, self.filters_blocks)
            for i, (old_features, weights, filter) in enumerate(source):
                patch = torch.nonzero(filter.data > 0).squeeze()
                new_bias = None
                if len(patch) == 0 or len(weights) == 0:
                    empty = Parameter(torch.zeros(0))
                    new_weights = empty
                    new_filter = empty
                    new_feature_ids.append(empty.data.long())
                    remove_parameter(self.optimizer, weights)
                    remove_parameter(self.optimizer, filter)
                    if self.has_bias:
                        new_bias = empty
                        remove_parameter(self.optimizer, self.bias_blocks[i])
                else:
                    new_feature_ids.append(old_features[patch.cpu()])
                    new_weights = Parameter(weights.data[patch])
                    new_filter = Parameter(filter.data[patch])
                    apply_patch(self.optimizer, weights, patch)
                    update_reference(self.optimizer, weights, new_weights)
                    apply_patch(self.optimizer, filter, patch)
                    update_reference(self.optimizer, filter, new_filter)
                    if self.has_bias:
                        bias = self.bias_blocks[i]
                        new_bias = Parameter(bias.data[patch])
                        apply_patch(self.optimizer, bias, patch)
                        update_reference(self.optimizer, bias, new_bias)
                new_weight_blocs.append(new_weights)
                new_filters_blocks.append(new_filter)
                if new_bias is not None:
                    new_bias_blocks.append(new_bias)

            self.out_features_map = new_feature_ids
            self.filters_blocks = new_filters_blocks
            self.weight_blocks = new_weight_blocs
            if hasattr(self, 'bias_blocks'):
                self.bias_blocks = new_bias_blocks

        self.regenerate_out_feature_ids()
        self.collecting = False

    def compute(self, x):
        if len(self.weight_blocks) == 0:
            raise AssertionError('Empty Model, call model.grow(size)')

        # Generate input tensors
        inputs = []
        if len(self.in_features_map) == 0:
            inputs = [x] * len(self.weight_blocks)
        else:
            start_idx = 0
            for features in self.in_features_map:
                if len(features) == 0:
                    inputs.append(self.wrap(torch.zeros(0)))
                else:
                    end_idx = start_idx + len(features)
                    inputs.append(x[:, start_idx:end_idx])
                    if not self.reuse_features:
                        start_idx += len(features)

        # Process the inputs
        results = []
        for i, (inp, weights) in enumerate(zip(inputs, self.weight_blocks)):
            bias = None
            if self.has_bias:
                bias = self.bias_blocks[i]
            if len(weights) != 0:
                result = self.compute_block(inp, weights, bias)
                if hasattr(self, 'filters_blocks'): # Apply filter if needed
                    filter = self.filters_blocks[i]
                    last_dim = len(result.size()) - 1
                    result = result.transpose(1, last_dim)
                    result = result * relu(filter)
                    result = result.transpose(1, last_dim)
                results.append(result)

        # Merge the results properly
        if len(results) == 0:
            raise AssertionError('Empty Model, call model.grow(size)')
        elif self.out_features is None:
            return torch.cat(results, dim=1)
        else:
            return torch.stack(results, dim=0).sum(0)

    def compute_block(self, x, weights, bias=None):
        raise NotImplementedError('Should be implemented by subclasses')

    def loss_factor(self):
        return float(np.array(self.weight_allocation).prod())

    @property
    def individual_filters(self):
        return [relu(x) for x in self.filters_blocks if len(x) > 0]

    def full_filter(self):
        individual_filters = self.individual_filters
        if len(individual_filters) == 0:
            return Variable(self.wrap(torch.zeros(0)))
        return torch.cat(individual_filters)

    def l1_loss(self, last_block=False):
        if hasattr(self, 'filters_blocks') and len(self.filters_blocks) > 0:
            if last_block:
                filter = self.filters_blocks[-1]
            else:
                filter = self.full_filter()
            return relu(filter).sum() * self.loss_factor()
        return Variable(self.wrap(torch.zeros(1)), requires_grad=False)

    @property
    def block_features(self):
        return [(x.data > 0).long().sum() for x in self.individual_filters]

    @property
    def num_output_features(self):
        if self.out_features is not None:
            return self.out_features
        if len(self.filters_blocks) == 0:
            return 0
        return (self.full_filter().data > 0).long().sum()

    @property
    def num_input_features(self):
        if self.in_features is not None:
            return self.in_features
        if len(self.in_features_map) == 0:
            return 0
        return len(set().union(*(set(x.numpy()) for x in self.in_features_map)))

    def generate_input(self):
        if self.in_features is None:
            raise ValueError('fake pass needs input or in_feature defined')
        size = (1, self.in_features) + self.additional_dims
        x = torch.rand(*size)
        x = torch.autograd.Variable(x, requires_grad=False)
        return self.wrap(x)

    @property
    def current_dimension_repr(self):
        return " [%s -> %s]" % (self.num_input_features, self.num_output_features)

class Linear(WeightedDynamicModule):

    def __init__(self, in_features=None, out_features=None, bias=True, weight_initializer=default_initializer, bias_initializer=default_initializer, reuse_features=True):
        super(Linear, self).__init__(
            in_features=in_features,
            out_features=out_features,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
            weight_allocation=(),
            bias=bias,
            reuse_features=reuse_features
        )

    def __repr__(self):
        inputs = '?' if self.in_features == None else self.in_features
        outputs = '?' if self.out_features == None else self.out_features
        return "Linear* (%s -> %s, bias=%s)" % (inputs, outputs, self.has_bias) + self.current_dimension_repr

    def compute_block(self,x, weights, bias):
        return linear(x, weights, bias)

class Conv2d(WeightedDynamicModule):
    def __init__(self, kernel_size,stride=1, in_channels=None, out_channels=None,
                 padding=0, dilation=1, groups=1, bias=True, reuse_features=True,
                 weight_initializer=default_initializer,
                 bias_initializer=default_initializer
                 ):

        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.kernel_size = _pair(kernel_size)
        self.output_padding = _pair(0)
        self.bias = bias

        super(Conv2d, self).__init__(
            in_features=in_channels,
            out_features=out_channels,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
            weight_allocation=self.kernel_size,
            reuse_features=reuse_features,
            bias=bias
        )

    def compute_dims(self):
        self.in_channels = self.in_features if self.in_features is not None else '?'
        self.out_channels = self.out_features if self.out_features is not None else '?'

    def compute_block(self, x, weights, bias):
        return conv2d(x, weights, bias, self.stride, self.padding, self.dilation, self.groups)

    def __repr__(self):
        self.compute_dims()
        return torch.nn.Conv2d.__repr__(self) + self.current_dimension_repr

class Flatten(DynamicModule):

    def __init__(self):
        super(Flatten, self).__init__()
        self._out_feature_ids = None
        self._in_feature_ids = None

    def collect_now(self, feature_ids, input_shape):
        if not self.collecting:
            return
        self.reset_features(feature_ids, input_shape)

    def grow_now(self, feature_ids, input_shape):
        if self.growing is False:
            return
        self.reset_features(feature_ids, input_shape)

    def reset_features(self, feature_ids, input_shape):
        self.in_features_map = feature_ids.clone().long()
        stride = int(np.array(input_shape[2:]).prod())
        r = torch.arange(0, stride).long()
        a = self.in_features_map * stride
        self._out_feature_ids = (r.unsqueeze(0).repeat(self.in_features_map.size(0), 1).transpose(0, 1) + a).transpose(0, 1).contiguous().view(-1)


    def compute(self, x):
        if self._out_feature_ids is None:
            raise AssertionError('empty model, call grow()')
        return x.view(x.size(0), -1)

    def __repr__(self):
        return 'Flatten()'

if __name__ == '__main__':
    data = Variable(torch.randn(2, 1, 50, 50))
    l1 = Conv2d(5, in_channels=1)
    l1.grow(100)
    l2 = Conv2d(5)
    l2.grow(100)
    flatten = Flatten()
    flatten.grow()
    l3 = Linear(out_features=10)
    l3.grow(1000)
    model = nn.Sequential(
        l1,
        nn.MaxPool2d(2),
        l2,
        nn.MaxPool2d(2),
        flatten,
        l3
    )
