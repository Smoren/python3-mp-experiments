import numba as nb
import numpy as np


@nb.cfunc("boolean[:](int64[:], int64[:])")
def add(where, what):
    where_size = where.shape[0]
    what_size = what.shape[0]
    result = np.empty(shape=(where_size,), dtype=np.bool_)
    for i in nb.prange(where_size):
        result[i] = False

    if what_size == 0:
        return result

    what_min, what_max = what[0], what[0]
    for i in nb.prange(1, what_size):
        if what[i] > what_max:
            what_max = what[i]
        elif what[i] < what_min:
            what_min = what[i]

    what_range = what_max - what_min

    what_normalized = np.empty(shape=(what_size + 1,), dtype=np.int64)
    for i in nb.prange(what_size):
        what_normalized[i] = what[i] - what_min

    isin_helper_ar = np.empty(shape=(what_range + 1,), dtype=np.int64)
    for i in nb.prange(what_range + 1):
        isin_helper_ar[i] = False
    for i in nb.prange(what_size):
        isin_helper_ar[what_normalized[i]] = True

    for i in nb.prange(where_size):
        if where[i] > what_max or where[i] < what_min:
            continue
        result[i] = isin_helper_ar[where[i] - what_min]

    return result


# @nb.cfunc("boolean[:](float64[:], float64[:])")
# def add(x, y):
#     return x + y


print(add.ctypes(np.ctypeslib.ndpointer(np.array([4])), np.ctypeslib.ndpointer(np.array([4]))))  # prints "9.0"
