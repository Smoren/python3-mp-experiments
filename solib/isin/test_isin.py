import time

import numpy as np
from ctypes import CDLL, c_int64, c_bool, c_int64
from numpy.ctypeslib import ndpointer

isin_lib = CDLL('./isin.so')

isin_lib.isin.argtypes = [
    ndpointer(c_int64, flags="C_CONTIGUOUS"),
    c_int64,
    ndpointer(c_int64, flags="C_CONTIGUOUS"),
    c_int64,
]


def isin_c(where: np.ndarray, what: np.ndarray) -> np.ndarray:
    isin_lib.isin.restype = ndpointer(c_bool, shape=(where.shape[0],))
    return isin_lib.isin(where, where.shape[0], what, what.shape[0])


def in1d(ar1, ar2):
    if ar2.size == 0:
        return np.zeros(shape=ar1.shape, dtype=np.bool_)

    # Convert booleans to uint8 so we can use the fast integer algorithm
    ar2_min = np.min(ar2)
    ar2_max = np.max(ar2)

    ar2_range = int(ar2_max) - int(ar2_min)
    outgoing_array = np.zeros(shape=a.shape, dtype=np.bool_)

    isin_helper_ar = np.zeros(ar2_range + 1, dtype=np.bool_)
    isin_helper_ar[ar2 - ar2_min] = 1

    # Mask out elements we know won't work
    basic_mask = (ar1 <= ar2_max) & (ar1 >= ar2_min)
    outgoing_array[basic_mask] = isin_helper_ar[ar1[basic_mask] - ar2_min]

    return outgoing_array


a = np.array(np.arange(10, 100000000, 2), dtype=np.int64)
b = np.array(np.arange(10, 100000000, 3), dtype=np.int64)

ts = time.time()
r1 = np.isin(a, b)
print(r1.shape, r1[:12])
print(f'np.isin(): {time.time() - ts}')

ts = time.time()
r2 = isin_c(a, b)
print(r2.shape, r2[:12])
print(f'isin_lib.isin(): {time.time() - ts}')

ts = time.time()
r3 = in1d(a, b)
print(r3.shape, r3[:12])
print(f'in1d(): {time.time() - ts}')

assert np.unique(r1 == r2) == np.array([True])

a = np.array(np.random.randint(-1000, 1000, 100000), dtype=np.int64)
b = np.array(np.random.randint(-1000, 1000, 100000), dtype=np.int64)
r1 = np.isin(a, b)
r2 = isin_c(a, b)
assert np.unique(r1 == r2) == np.array([True])

a = np.array(np.random.randint(-2000, 2000, 100000), dtype=np.int64)
b = np.array(np.random.randint(-1000, 1000, 100000), dtype=np.int64)
r1 = np.isin(a, b)
r2 = isin_c(a, b)
assert np.unique(r1 == r2) == np.array([True])

a = np.array(np.random.randint(-1000, 1000, 100000), dtype=np.int64)
b = np.array(np.random.randint(-2000, 2000, 100000), dtype=np.int64)
r1 = np.isin(a, b)
r2 = isin_c(a, b)
assert np.unique(r1 == r2) == np.array([True])

a = np.array(np.random.randint(-1000, -100, 100000), dtype=np.int64)
b = np.array(np.random.randint(100, 2000, 100000), dtype=np.int64)
r1 = np.isin(a, b)
r2 = isin_c(a, b)
assert np.unique(r1 == r2) == np.array([True])

a = np.array(np.arange(0, 100), dtype=np.int64)
b = np.array(np.arange(99, 200), dtype=np.int64)
r1 = np.isin(a, b)
r2 = isin_c(a, b)
assert np.unique(r1 == r2) == np.array([True])

print('OK')