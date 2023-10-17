import multiprocessing as mp
import numpy as np
import math

from common import ARRAY_SIZE, NUM_WORKERS, with_profiling, DATA_LHS, DATA_RHS


def np_sum(lhs, rhs):
    return np.sum(lhs + rhs)


@with_profiling
def benchmark():
    chunk_size = int(math.ceil(ARRAY_SIZE / NUM_WORKERS))

    with mp.Pool(NUM_WORKERS) as pool:
        task_data = [(
            DATA_LHS[i * chunk_size:i * chunk_size + chunk_size],
            DATA_RHS[i * chunk_size:i * chunk_size + chunk_size],
        ) for i in range(NUM_WORKERS)]
        results = pool.starmap(np_sum, task_data)

    return np.sum(results)


benchmark()
