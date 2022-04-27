import itertools
from typing import Iterable

import numpy as np
from sklearn import metrics


def kron_combine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    kron(A, 11^T) + kron(11^T, B)
    Note: This is not the same as a Kronecker sum, which uses identities in place of the ones matrices.
    """
    return np.kron(A, np.ones(B.shape)) + np.kron(np.ones(A.shape), B)

def tuple_id(t: Iterable[int], levels: Iterable[int]) -> int:
    """
    A bijection that maps a tuple of ints to a scalar integer
    Example with levels = (2, 3):
    (0, 0) -> 0
    (0, 1) -> 1
    (0, 2) -> 2
    (1, 0) -> 3
    (1, 1) -> 4
    (1, 2) -> 5
    """
    values = list(t)
    multiplier = 0
    for i in reversed(range(len(values))):
        if multiplier > 0:
            values[i] *= multiplier
        multiplier += levels[i]
    
    return sum(values)

def from_tuple_id(id: int, levels: Iterable[int]) -> tuple[int]:
    """
    Inverse of tuple_id
    """
    entries = []
    for i in reversed(range(len(levels))):
        remainder = id % levels[i]
        entries.append(remainder)
        id = id // levels[i]
    return tuple(reversed(entries))

def label_accuracy(labels: Iterable[int], truth: Iterable[int]) -> float:
    """
    max(accuracy) over the set of all label permutations
    truth should be an list of 0-indexed integer labels of length n
    """
    accuracy = 0
    k = max(truth) + 1 # number of labels
    
    # This is not optimal, but we're using small k, so it's no biggie.
    for p in itertools.permutations(range(k)):
        compare = [p[t] for t in truth]
        accuracy = max(accuracy, metrics.accuracy_score(labels, compare))
    
    return accuracy
