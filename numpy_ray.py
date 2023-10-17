import ray
import numpy as np
import math

from common import NUM_WORKERS, ARRAY_SIZE, DATA_RHS, DATA_LHS, with_profiling

# ray init can take 3 seconds or more to load
ray.init(num_cpus=NUM_WORKERS)


@ray.remote
def np_sum_ray2(lhs, rhs, start, stop):
    return np.sum(lhs[start:stop] + rhs[start:stop])


@with_profiling
def benchmark():
    chunk_size = int(math.ceil(ARRAY_SIZE / NUM_WORKERS))
    futures = []
    lhs_ref = ray.put(DATA_LHS)
    rhs_ref = ray.put(DATA_RHS)

    for i in range(0, NUM_WORKERS):
        start = i * chunk_size
        futures.append(np_sum_ray2.remote(lhs_ref, rhs_ref, start, start + chunk_size))
    return np.sum(ray.get(futures))


benchmark()
