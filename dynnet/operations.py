from dynnet.interfaces import TensorOperation

from torch import Tensor, LongTensor


class IndexSelectOperation(TensorOperation):
    """Operation that selects only the given indices on a given dimension"""

    def __init__(self, indices: LongTensor, dimension: int = 0):
        """Create the operation

        Parameters
        ----------
        indices
            The indices to select from the tensor
        dimension
            The dimension to select on
        """
        self.indices: LongTensor = indices
        self.dimension: int = dimension

    def __call__(self, tensor: Tensor):
        try:
            assert tensor.size(self.dimension) > self.indices.size(0), (
                "The operation should remove at least one feature")
        except AssertionError:
            print(tensor, self.indices)
            raise
        if self.dimension != 0:
            tensor = tensor.transpose(self.dimension, 0)
        if tensor.is_cuda:
            indices = self.indices.cuda(tensor.get_device())
        else:
            indices = self.indices
        tensor = tensor[indices]
        if self.dimension != 0:
            tensor = tensor.transpose(0, self.dimension)
        return tensor
