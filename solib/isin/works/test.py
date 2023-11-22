import numpy as np
import numba as nb
import cffi
from numba.core.typing import cffi_utils

import cffi_isin

ffi = cffi.FFI()
cffi_utils.register_module(cffi_isin)
isin = cffi_isin.lib.isin


@nb.njit
def test(where, what):
    where_size, what_size = where.shape[0], what.shape[0]
    p_where_size, p_what_size = np.int64(where_size), np.int64(what_size)
    p_where, p_what = ffi.from_buffer(where.astype(np.int64)), ffi.from_buffer(what.astype(np.int64))

    result = np.empty(where_size, dtype=np.bool_)
    p_result = ffi.from_buffer(result.astype(np.bool_))
    isin(p_where, p_where_size, p_what, p_what_size, p_result)
    return nb.carray(p_result, result.shape[0], result.dtype)


where = np.array([1, 2, 3, 4, 5])
what = np.array([1, 3, 5])
r = test(where, what)
print(r)
