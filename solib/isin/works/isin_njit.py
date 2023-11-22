import time

import numpy as np
import numba as nb
import cffi
from numba.core.typing import cffi_utils

from solib.isin.works import cffi_isin

_ffi = cffi.FFI()
cffi_utils.register_module(cffi_isin)
_isin = cffi_isin.lib.isin


@nb.njit
def isin_cffi(where, what):
    where_size, what_size = where.shape[0], what.shape[0]
    print(where_size, what_size)
    p_where_size, p_what_size = np.int64(where_size), np.int64(what_size)
    p_where, p_what = _ffi.from_buffer(where.astype(np.int64)), _ffi.from_buffer(what.astype(np.int64))

    result = np.empty(where_size, dtype=np.bool_)
    p_result = _ffi.from_buffer(result.astype(np.bool_))
    _isin(p_where, p_where_size, p_what, p_what_size, p_result)
    return nb.carray(p_result, result.shape[0], result.dtype)


if __name__ == "__main__":
    # where = np.array([1, 2, 3, 4, 5])
    # what = np.array([1, 3, 5])
    # r = isin_cffi(where, what)
    # print(r)

    ts = time.time()
    a = np.array(np.arange(10, 900, 2), dtype=np.int64)
    b = np.array(np.arange(10, 900, 3), dtype=np.int64)
    r = isin_cffi(a, b)
    print(r.shape, r[:12])
    print(f'isin_cffi() (first): {time.time() - ts}')
    print()

    ts = time.time()
    a = np.array(np.arange(10, 900, 2), dtype=np.int64)
    b = np.array(np.arange(10, 900, 3), dtype=np.int64)
    r = isin_cffi(a, b)
    print(r.shape, r[:12])
    print(f'isin_cffi() (first): {time.time() - ts}')
