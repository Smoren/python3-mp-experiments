import time

import numpy as np
import numba as nb
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


@nb.njit(
    fastmath=True,
    boundscheck=False,
    looplift=True,
    nogil=True,
)
def in1d(ar1, ar2):
    # Ravel both arrays, behavior for the first array could be different
    # ar1 = np.asarray(ar1).ravel()
    # ar2 = np.asarray(ar2).ravel()

    if ar2.size == 0:
        return np.zeros(shape=ar1.shape, dtype=np.bool_)

    # Convert booleans to uint8 so we can use the fast integer algorithm
    ar2_min = np.min(ar2)
    ar2_max = np.max(ar2)

    ar2_range = int(ar2_max) - int(ar2_min)

    # Optimal performance is for approximately
    # log10(size) > (log10(range) - 2.27) / 0.927.
    # However, here we set the requirement that by default
    # the intermediate array can only be 6x
    # the combined memory allocation of the original
    # arrays. See discussion on
    # https://github.com/numpy/numpy/pull/12065.
    outgoing_array = np.zeros(shape=a.shape, dtype=np.bool_)
    # outgoing_array = np.zeros(shape=a.shape, dtype=bool)
    # outgoing_array = np.zeros_like(ar1, dtype=bool)

    isin_helper_ar = np.zeros(ar2_range + 1, dtype=np.bool_)
    isin_helper_ar[ar2 - ar2_min] = 1

    # Mask out elements we know won't work
    basic_mask = (ar1 <= ar2_max) & (ar1 >= ar2_min)
    outgoing_array[basic_mask] = isin_helper_ar[ar1[basic_mask] - ar2_min]

    return outgoing_array


a = np.array(np.arange(0, 10000000, 2), dtype=np.int32)
b = np.array(np.arange(0, 10000000, 3), dtype=np.int32)

# ts = time.time()
# r = np.isin(a, b)
# print(r.shape, r[:12])
# print(f'np.isin(): {time.time() - ts}')
#
# ts = time.time()
# r = isin_c(a, b)
# print(r.shape, r[:12])
# print(f'isin_lib.isin(): {time.time() - ts}')

ts = time.time()
r = in1d(a, b)
print(r.shape, r[:12])
print(f'in1d(): {time.time() - ts}')
