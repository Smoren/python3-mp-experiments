import time

import numpy as np
from ctypes import CDLL, c_int32, c_bool
from numpy.ctypeslib import ndpointer

isin_lib = CDLL('./isin.so')

isin_lib.isin.argtypes = [
    ndpointer(c_int32, flags="C_CONTIGUOUS"),
    c_int32,
    ndpointer(c_int32, flags="C_CONTIGUOUS"),
    c_int32,
]


def isin_c(where: np.ndarray, what: np.ndarray) -> np.ndarray:
    isin_lib.isin.restype = ndpointer(c_bool, shape=(where.shape[0],))
    return isin_lib.isin(where, where.shape[0], what, what.shape[0])


a = np.array(np.arange(0, 10000000, 2), dtype=np.int32)
b = np.array(np.arange(0, 10000000, 3), dtype=np.int32)

ts = time.time()
r = np.isin(a, b)
print(r.shape, r[:12])
print(f'np.isin(): {time.time() - ts}')

ts = time.time()
r = isin_c(a, b)
print(r.shape, r[:12])
print(f'isin_lib.isin(): {time.time() - ts}')
