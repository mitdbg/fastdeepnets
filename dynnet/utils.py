"""Some handy independent functions"""

from typing import List, Union

import numpy as np
from torch import from_numpy, LongTensor


def is_sorted(lst: List) -> bool:
    """Check whether a iterable is sorted (ascending)

    Taken from https://stackoverflow.com/a/4404056

    Parameters
    ----------
    lst
        The list we want to check

    Returns
    -------
    :True: if the list is sorted in ascending order, :False: otherwise
    """
    for i, element in enumerate(lst[1:]):
        if element <= lst[i-1]:
            return False
    return True


def compute_feature_patch(previous_features: List,
                          new_features: List,
                          as_tensor: bool = False) -> Union[List,
                                                            LongTensor]:
    """Compute a patch to go from the previous features to the new ones

    Parameters
    ----------

    previous_features
        The list of previous features
    new_features
        The list of new features

    Returns
    -------

    The indices to pick from the :previous: to yield the :new_features:
    """
    assert len(new_features) <= len(previous_features), (
        "We can only remove features")
    result = [previous_features.index(x) for x in new_features]
    if as_tensor:
        result = from_numpy(np.array(result))
    return result

def set_to_ordered_list(features_set: set) -> List:
    """Converts a set to an ordered numpy array

    Parameters
    ----------

    features_set
        the set of comparable elements


    Returns
    -------

    A sorted numpy array
    """
    return sorted(list(features_set))


def sorted_list_intersection(arr1: List, arr2: List) -> List:
    """Compute the intersection of two set represented by sorted lists

    Parameters
    ----------
    arr1
        The first set (represented as a sorted list)
    arr2
        The second set (represented as a sorted list)

    Returns
    -------
    A new sorted list (as an np.array) containing the intersection
    of the two input lists
    """
    # We make sure the smaller array is in arr1
    # We will iterate on the smallest array first because it is a property
    # of the intersection that the resulting set cannot be bigger than the
    # smallest one
    if len(arr1) > len(arr2):
        arr2, arr1 = arr1, arr2

    result = []
    p2 = 0
    for v1 in arr1:
        while p2 < len(arr2) and arr2[p2] < v1:
            p2 += 1
        if p2 >= len(arr2):  # We will not find any match anymore
            break
        if arr2[p2] == v1:
            result.append(v1)
            # We can already increase p2 because we suppose that there is not
            # duplicate in any of the two input list (they are supposed to
            # represent sets after all
            p2 += 1
    return result


def batch_indexing(array: List, indices: List) -> List:
    """Index all offsets of :indices: into the list :array:

    Parameters
    ----------
    array:
        The array containing the values
    indices:
        The positions of the values we are interested in

    Returns
    -------
    The values of of :array: corresponding to the indices found in :indices:
    """
    return [array[x] for x in indices]
