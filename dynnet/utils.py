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
