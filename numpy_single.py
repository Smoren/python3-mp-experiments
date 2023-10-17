import numpy as np

from common import DATA_LHS, DATA_RHS, with_profiling


@with_profiling
def benchmark():
    return np.sum(DATA_LHS + DATA_RHS)


benchmark()
