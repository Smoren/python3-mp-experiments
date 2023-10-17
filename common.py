# TODO more variants: https://luis-sena.medium.com/sharing-big-numpy-arrays-across-python-processes-abf0dc2a0ab2

import multiprocessing as mp
import time
import numpy as np

np.random.seed(42)

NUM_WORKERS = mp.cpu_count()
ARRAY_SIZE = int(4e8 + 113)
DATA_LHS = np.random.random(ARRAY_SIZE)
DATA_RHS = np.random.random(ARRAY_SIZE)


def with_profiling(func):
    def wrapper():
        ts = time.time()
        print('started')
        result = func()
        print('finished', round(time.time() - ts, 4), result)
    return wrapper
