"""Implementation of a computation graph on top of pytorch"""
from collections import defaultdict, deque
from typing import Dict, List, Callable, Optional, Union, Any
from torch.nn import ModuleList, Module

from dynnet.interfaces import (DynamicModule,
                               GarbageCollectionLog)
import dynnet.layers


class Graph(ModuleList):
    """
    Represent a computation graph

    This is a necessary addition to pytorch because we need the parent/child
    relationshipt in order to do proper garbage collection

    Attributes
    ----------

    _children
        A map that gives the children of a given module

    parents
        A map that gives the parents of a given module
    """

    def __init__(self):
        super(Graph, self).__init__()
        self._children: Dict[int,
                            List[int]] = defaultdict(list)
        self.parents: Dict[int,
                           List[int]] = defaultdict(list)

    def add(self, factory: Callable[[], Module],
            *args, **kwargs) -> Callable[[Optional[List[DynamicModule]]],
                                         DynamicModule]:
        """Utilitary function to quickly build a simple graph

        Prepare the creation of a Dynamic module

        Parameters
        ----------
        factory
            The constructor for the module
        args
            arguments for the constructor
        kwargs
            keyword arguments for the constructor

        Returns
        -------
        the module boxed into a graph vertex with links set up
        """
        def create(*parents: List[DynamicModule]) -> DynamicModule:
            """Function that given the parents constructs the module

            It also set up the graph links apropriately

            Parameters
            ----------

            parents
                The potential parents of the module
            """
            # Gathering input features
            input_features = []
            if parents:
                for parent in parents:
                    if parent.output_features is not None:
                        input_features.append(parent.output_features)

            # Construct the module with info about the features
            kwargs['input_features'] = input_features
            kwargs['graph'] = self
            if not issubclass(factory, DynamicModule):
                real_args = (factory,) + args
                real_factory = dynnet.layers.NaiveWrapper
            else:
                # Avoid writing the factory and args variables
                # Because you definitely do not want a global keyword here
                real_factory = factory
                real_args = args

            module = real_factory(*real_args, **kwargs)
            self.append(module)

            # Setting up the graph
            if parents is not None:
                self.parents[self.get_index(module)] = [self.get_index(x) for x in parents]
                for parent in parents:
                    self._children[self.get_index(parent)].append(self.get_index(module))
            return module
        return create

    def resolve_index(self, index):
        return self[index]

    def get_index(self, module):
        for i, other_module in enumerate(self):
            if other_module == module:
                return i
        raise Exception("Module %s was not found" % module)

    def garbage_collect(self):
        """Triggers garbage collection in the graph
        Returns
        -------

        The log of operations that were applied during the garbage
        collection process
        """
        log = GarbageCollectionLog()

        for module in self:
            module.garbage_collect(log)

        return log

    def forward(self, input_map: Dict[Module, List[Any]],
                *outputs: List[Module]):
        """Forward through this graph
        This method is sligthly different than normal pytorch forward function
        You give a map of module and tensors they should take as inputs and
        also specifies the modules you want output from and the graph will
        figure out which module to run with which inputs to answer the query

        Parameters
        ----------

        input_map
            The map going from module to whaterver you want them to swallow
        outputs
            The list of modules you want the output from

        Returns
        -------

        A map going from each module from the :outputs: parameter to the the
        value they output. The Graph will figure out the dependencies for all
        your modules and how to obtain them recursively

        Raises
        ------

        AssertionError
            Will raise AssertionError if there is no way to resolve the
            dependencies

        Internal Note
        -------------
        In this implementation we might be keeping too much memory, it would
        be useful to remove any tensor from memory if we know we will never
        """
        memory = {self.get_index(k): v for k, v in input_map.items()}

        outputs = list(outputs)
        outputs = [self.get_index(x) for x in outputs]
        requested_outputs = outputs[:]

        # Simulating a stack to avoid stack overflows and computing an output
        # multiple times
        while outputs:
            module_id = outputs.pop()
            parents = self.parents[module_id]
            assert parents, (
                "Cannot compute without an input for %s" %
                self.resolve_index(module_id))
            # Computing all the parents that do not have their output
            # available. A very pythonic way to do it would have been
            # to catch a key Error but we would not have been able to
            # populate the stack with all the missing module (only the
            # first one
            left_to_do = [p for p in parents if p not in memory]
            if not left_to_do:
                result = self.resolve_index(module_id)(*[memory[p] for p in parents])
                memory[module_id] = result
            else:
                outputs.append(module_id)
                outputs.extend(left_to_do)

        return [memory[m] for m in requested_outputs]

class Sequential(Graph):

    def __init__(self):
        super(Sequential, self).__init__()

    def add(self, factory: Callable[[], Module],
            *args, **kwargs) -> DynamicModule:
        parent = []
        if len(self) > 0:
            parent = [self[-1]]
        return super(Sequential, self).add(factory, *args, **kwargs)(*parent)

    def forward(self, inp):
        return super(Sequential, self).forward({self[0]: inp}, self[-1])[0]
