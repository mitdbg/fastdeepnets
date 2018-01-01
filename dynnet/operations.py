from dynnet.interfaces import TensorOperation

from torch import Tensor, LongTensor


class IndexSelectOperation(TensorOperation):
    """Operation that selects only the given indices on a given dimension"""

    def __init__(self, indices: LongTensor, dimension: int=0):
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
        tensor_sum = tensor.long().sum()
        assert tensor.size(0) > tensor_sum, (
            "The operation should remove at least one feature")
        if self.dimension != 0:
            tensor = tensor.transpose(self.dimension, 0)
        if tensor.is_cuda:
            print('start_cuda')
            indices = self.indices.cuda(tensor.get_device())
            print('end_cuda')
        else:
            indices = self.indices
        print('apply')
        tensor = tensor[indices]
        if self.dimension != 0:
            tensor = tensor.transpose(0, self.dimension)
        return tensor
