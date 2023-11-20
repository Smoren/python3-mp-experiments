import time

import numpy as np
import numba as nb
from ctypes import CDLL, c_bool, c_int64
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


@nb.njit(looplift=True, nogil=True)
def isin_py(where: np.ndarray, what: np.ndarray) -> np.ndarray:
    where_size = where.shape[0]
    what_size = what.shape[0]
    result = np.empty(shape=(where_size,), dtype=np.bool_)
    for i in nb.prange(what_size):
        result[i] = False

    if what_size == 0:
        return result

    what_min, what_max = what[0], what[0]
    for i in nb.prange(what_size):
        if what[i] > what_max:
            what_max = what[i]
        elif what[i] < what_min:
            what_min = what[i]

    what_range = what_max - what_min

    what_normalized = np.empty(shape=(what_size+1,), dtype=np.int64)
    for i in nb.prange(what_size):
        what_normalized[i] = what[i] - what_min

    isin_helper_ar = np.empty(shape=(what_range+1,), dtype=np.int64)
    for i in nb.prange(what_range):
        isin_helper_ar[i] = False
    for i in nb.prange(what_size):
        isin_helper_ar[what_normalized[i]] = True

    for i in nb.prange(where_size):
        if where[i] > what_max or where[i] < what_min:
            continue
        result[i] = isin_helper_ar[where[i] - what_min]

    return result


a = np.array(np.arange(10, 100000000, 2), dtype=np.int64)
b = np.array(np.arange(10, 100000000, 3), dtype=np.int64)

ts = time.time()
r2 = isin_py(a, b)
print(r2.shape, r2[:12])
print(f'isin_py(): {time.time() - ts}')

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