"""All the interfaces related to the dynamic neural networks"""

from typing import List, Any, Callable, Tuple
from collections import namedtuple
import numpy as np
from torch import Tensor, LongTensor, is_tensor
from torch.nn import Module, Parameter
from torch.optim import Optimizer

from dynnet.utils import (compute_feature_patch,
                          sorted_list_intersection,
                          batch_indexing)

GarbageCollectionOperation = namedtuple('GarbageCollectionOperation',
                                        ['old_parameter',
                                         'operation',
                                         'new_parameter'])


class TensorOperation:
    """Represents a repeatable operation (out of place) on a tensor"""

    def __call__(self, tensor: Tensor) -> Tensor:
        """Apply the operation on a tensor

        Parameters
        ----------
        tensor
            The tensor to apply the operation on

        Returns
            A new tensor with the operation applied
        """
        raise NotImplementedError("This is a virtual class")


class GarbageCollectionLog():
    """This class contains a log of what happened during garbage
    collection. It is especially usefull to update optimizers
    to continue training after garbage collection

    Attributes
    ----------
    operations: List
        The list of operations that were logged (in order)
    """

    def __init__(self):
        self.operations: List = []

    def change_parameter(self, parameter: Parameter,
                         operation: TensorOperation) -> Parameter:
        """Apply and log the opration on the given tensor

        the operation will be applied on the tensor and remebered.
        It will allow us to apply the same opration to keep the state
        of the optimizers up to date.

        Parameters
        ----------
        parameter
            the parameter to modify
        operation
            the operation to apply and log

        Returns
        -------

        the tensor modified in a new parameter
        """
        result = Parameter(operation(parameter.data))
        self.operations.append(GarbageCollectionOperation(
            parameter, operation, result))
        return result

    def update_optimizer(self, optimizer: Optimizer) -> None:
        """
        Update an optimizer to make it compatible with the parameters
        after the garbage collection logged were applied

        Parameters
        ----------
        optimizer
            The optimizer to update
        """
        opt_state = optimizer.state
        for ope in self.operations:
            # 1 - Updating the param groups
            for group in optimizer.param_groups:
                group['params'] = [ope.new_parameter if x is ope.old_parameter
                                   else x
                                   for x in group['params']]

            # 2 - Udating the paramter state
            state = opt_state[ope.old_parameter]
            next_state = {}
            for key, tensor in state.items():
                # Make sure we update tensors with same size
                # If not there is nothing we can do
                # If needed we might want to allow custom behavior
                # To handle weird optimizers
                if (is_tensor(tensor)
                        and tensor.size() == ope.old_parameter.size()):
                    next_state[key] = ope.operation(tensor)
                else:
                    next_state[key] = tensor
            # This paramter does not exist anymore
            # We can get rid of it
            del opt_state[ope.old_parameter]
            # Adding back the new parameter
            opt_state[ope.new_parameter] = next_state


class FeatureBag():
    """Handle synchronization of features between modules

    Internal Note
    -------------

    We could maybe save some memory by only maintaining the :module_awareness:
    dict during the event propagation process, otherwise they should all be
    equal to :self.latest_features:

    Attributes
    ----------

    latest_features
        The set of the list of features in this bag
    input_listeners
        The set of modules that uses this bag of feature as an input
    output_listeners
        The set of muldules that uses this bag of feature as an output
        (It should actually be a single module but in very weird situations
        it might make sense to have more)
    module_awareness
        The map from module to the set of feature this module THINKS is in the
        bag
    input_indices
        Each module that uses this bag as an input might have multiple input,
        therefore they need a way to know which bag of feature we talk about,
        this is why they pass an index when they register. This attribute store
        the way modules want to hear about this bag of feature
    propagating
        This is a flag that says if we are in the process of synchronizing and
        resolving the conflicts between the modules. During this phase it is
        not recommended to touch the internals of this object, and of course
        add listeners
    """
    def __init__(self, feature_count: int, *additional_dims: Tuple):
        self.additional_dims = additional_dims
        self.latest_features = list(range(0, feature_count))
        self.input_listeners = set()
        self.output_listeners = set()
        self.module_awareness = dict()
        self.input_indices = dict()
        self.propagating = False

    @property
    def feature_count(self):
        """Return the number of features in this FeatureBag

        It is unsafe to read this value during event propagation
        Some layers might not be aware of the current size yet
        Otherwise it is fine
        """
        return len(self.latest_features)

    def sample_typical_input(self, tpe: Callable = Tensor,
                             batch_size: int = 2) -> Tensor:
        """Generate an input that has a compatible size with this FeatureBag

        It is especially useful for layers that want to compute their output
        type. They can just get a sample, forward it and gather the ouput
        shape

        Parameters
        ----------
        type
            The constructor of the type of tensors you want to sample from
            The actual type will depend on pytorch configuration
        batch_size
            The batch size of the sampled tensor. By default it is two because
            it is the smallest unsqueezable size

        Returns
        -------
            A tensor of the correct size
        """
        return tpe(batch_size, self.feature_count, *self.additional_dims)

    def offsets_to_features(self, context: "DynamicModule",
                            offsets: List):
        """Converts the feature offsets of a module to a set of feature ids

        Parameters
        ----------
        context
            The module these offsets belong to
        offsets
            The offset we want to convert

        Returns
        -------

        A set of feature ids corresponding to these offsets
        """
        context_features = self.module_awareness[context]
        return batch_indexing(context_features, offsets)

    def remove_features(self, context: "DynamicModule",
                        remaining_offsets: LongTensor,
                        log: GarbageCollectionLog) -> None:
        """Remove features from this FeatureBag by specifing the ones to keep

        Parameters
        ----------

        context
            The module requesting to remove the features
        remaining_offsets
            The offsets (relative to the module) to keep
        log
            The garbage collector log to register any change made to the
            parameters
        """
        assert context in self.module_awareness, (
            "The module requesting the feature removal is not a listener")
        # We do not care about the torch format here
        remaining_offsets = remaining_offsets.cpu().numpy()

        # Compute the features from the offsets in context
        remaining_features = self.offsets_to_features(context,
                                                      remaining_offsets)
        if len(remaining_features) == len(self.latest_features):
            # We are effectively not modifing the features
            return
        # Remove the useless features
        self.latest_features = sorted_list_intersection(remaining_features,
                                                        self.latest_features)

        self.propagate_changes(log)

    def propagate_changes(self, log: GarbageCollectionLog) -> None:
        """Propagate the update to the feature set to all listeners

        Parameters
        ----------
        log
            The garbage collector log to register any change made to the
            parameters
        """
        if self.propagating:
            # Somone upper in the call stack is already doing it
            return
        self.propagating = True
        # We loop until every module is aware of the latest feature set
        # It should always eventually terminate because every time the
        # feature set changes we should remove at least one feature
        # and since the number of feature is finite it will terminate
        while True:
            for module, awareness in self.module_awareness.items():
                # It is VERY important here to directly get the feature
                # set from the self directly because we are not modifying
                # it in place
                if awareness != self.latest_features:
                    self.warn_module(module, log)
                    break
            else:
                break
        self.propagating = False

    @property
    def root_feature_bag(self):
        """Return the referene feature bag for a given feature bag

        For normal feature bags it is the instance itself
        """
        return self

    def warn_module(self, module: "DynamicModule",
                    log: GarbageCollectionLog) -> None:
        """Update a module with the latest features

        Parameters
        ----------
        module
            The module to warn
        log
            The garbage collector log to register any change made to the
            parameters
        """
        current_features = self.latest_features
        previous_features = self.module_awareness[module]
        patch = compute_feature_patch(previous_features, current_features,
                                      as_tensor=True)
        if module in self.input_listeners:
            # Following the FeatureBag chain until the root for all inputs
            roots = [x.root_feature_bag for x in module.input_features]
            index = roots.index(self)
            module.remove_input_features(patch, index, log)
        else:
            module.remove_output_features(patch, log)
        self.module_awareness[module] = self.latest_features

    def preregister(self, listener: "DynamicModule") -> None:
        """Common operations (input and output) for listener registration

        Parameters
        ----------

        listener
            The listener to register
        """
        assert not self.propagating, (
            "It is dangerous to add events during event processing")
        assert isinstance(listener, DynamicModule), (
            "Only DynamicModule can be warned of feature removal")
        self.module_awareness[listener] = self.latest_features[:]

    def register_input_listener(self, listener: 'DynamicModule',
                                input_index: Any) -> None:
        """Register a module that uses these features as an input

        Parameters
        ----------

        listener
            The module to register
        """
        self.preregister(listener)
        self.input_listeners.add(listener)
        self.input_indices[listener] = input_index

    def register_output_listener(self, listener: 'DynamicModule') -> None:
        """Register a module that uses these features as an output

        Parameters
        ----------

        listener
            The module to register
        """
        self.preregister(listener)
        self.output_listeners.add(listener)

    def __repr__(self):
        return "FeatureBag(%s, %s)" % (self.feature_count,
                                       self.additional_dims)


class MirrorFeatureBag(FeatureBag):
    """This is a FeatureBag linked to another
    The two feature bags share the same features but they might have different
    additional dimensions

    This is espacially useful for layers that only change the size of the
    additional dimensions (MaxPools) but do not change the number of
    channels

    MirrorFeatureBag can be chained, i.e. be linked to another MirrorFeatureBag

    Attributes
    ----------

    reference_feature_bag
        The feature_bag to mirror
    """

    def __init__(self, reference_feature_bag, *additional_dims):
        self.reference_feature_bag = reference_feature_bag
        self.additional_dims = additional_dims

    def register_output_listener(self, listener: 'DynamicModule') -> None:
        self.reference_feature_bag.register_output_listener(listener)

    def register_input_listener(self, listener: 'DynamicModule',
                                input_index: Any) -> None:
        self.reference_feature_bag.register_input_listener(
            listener, input_index
        )

    @property
    def root_feature_bag(self):
        """Return the referene feature bag for a given feature bag

        For MirrorFeatureBag we follow the chain until we reach a FeatureBag
        """
        return self.reference_feature_bag.root_feature_bag

    @property
    def feature_count(self) -> int:
        return self.reference_feature_bag.feature_count

    def remove_features(self, context: "DynamicModule",
                        remaining_offsets: LongTensor,
                        log: GarbageCollectionLog) -> None:
        self.reference_feature_bag.remove_features(
            context, remaining_offsets, log)

    def __repr__(self):
        return "MirrorFeatureBag(%s, %s)" % (self.feature_count,
                                             self.additional_dims)


class DynamicModule(Module):
    """The base class for any Dynamic layer

    This is an interface, any meaningful layer should reimplement
    the main methods

    Attributes
    ----------

    input_features
        A list of feature list provided by all its parents
    output_features
        The feature list outputed by this module
    """

    def __init__(self, input_features: List[FeatureBag],
                 output_features: FeatureBag, graph: "Graph" = None):
        super(DynamicModule, self).__init__()
        # We do not want to register this module
        # It may cause recursios errors in the __repr__ calls
        self.__dict__['graph'] = graph
        self.input_features: List[FeatureBag] = input_features
        self.output_features: FeatureBag = output_features

        # Register module to feature events
        self.output_features.register_output_listener(self)
        for input_index, input_feature_bag in enumerate(self.input_features):
            input_feature_bag.register_input_listener(self, input_index)

    def forward(self, *args):
        assert(len(args) == len(self.input_features)), (
            "Invalid number of arguments for forward %s (expected %s)" % (
                len(args), len(self.input_features)))
        for i, (arg, feature_bag) in enumerate(zip(args, self.input_features)):
            dims = arg.size()[1:]
            expected_dims = ((feature_bag.feature_count,)
                             + feature_bag.additional_dims)
            assert dims == expected_dims, (
                "Invalid dimensions for input %s, got %s (expected %s)" % (
                    i, dims, expected_dims))

    def garbage_collect(self, log: GarbageCollectionLog):
        """Trigger garbage collection of the module and computes the
        remaining useful features

        Parameters
        ----------
        log
            The log of garbage collection operations

        """
        # The default implementation does nothing
        pass

    def remove_input_features(self, remaining_features: LongTensor,
                              input_index: Any,
                              log: GarbageCollectionLog) -> None:
        """Ask a module to remove input features based on the ones to keep

        Parameters
        ----------
        remaining_features
            the features to keep. They are offsets to the current features
            used by this module
        input_index
            The index of the input we are affecting (some modules might have)
            multiple inputs
        log
            The log of garbage collection operations
        """
        # The default implementation does nothing
        pass

    def remove_output_features(self, remaining_features: LongTensor,
                               log: GarbageCollectionLog) -> None:
        """Ask a module to remove output features based on the ones to keep

        Parameters
        ----------
        remaining_features
            the features to keep. They are offsets to the current features
            used by this module
        log
            The log of garbage collection operations
        """
        # The default implementation does nothing
        pass
