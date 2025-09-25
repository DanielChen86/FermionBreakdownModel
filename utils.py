from openfermion import FermionOperator, normal_ordered
from typing import List
from itertools import combinations
import numpy as np


def normal_vacuum(operator: FermionOperator) -> FermionOperator:
    returned = []
    op = normal_ordered(operator)
    for k, v in op.terms.items():
        if k[-1][-1] == 1:
            returned.append(FermionOperator(k, v))
    if len(returned) == 0:
        return None
    else:
        return sum(returned)


def list_to_str(x: List[int]):
    return ','.join(map(str, x))


def H(n: int, r: int):
    if n == 1:
        return [[r]]
    returned = []
    N = n + r - 1
    for x in combinations(range(N), n-1):
        arr = np.array(x)
        arr_return = np.concatenate(
            (arr[0:1], (arr[1:] - arr[:-1] - 1), N-arr[-1:]-1))
        returned.append(arr_return.tolist())
    return returned
