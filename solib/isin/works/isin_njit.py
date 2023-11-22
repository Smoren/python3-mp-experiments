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
    where, what = where.astype(np.int64), what.astype(np.int64)
    where_size, what_size = where.shape[0], what.shape[0]
    p_where_size, p_what_size = np.int64(where_size), np.int64(what_size)
    p_where, p_what = _ffi.from_buffer(where), _ffi.from_buffer(what)

    p_result = _isin(p_where, p_where_size, p_what, p_what_size)
    return nb.carray(p_result, where.shape[0], np.bool_)


def _bench(a, b):
    ts = time.time()
    r = isin_cffi(a, b)
    print(r.shape, r[:12])
    print(f'isin_cffi(): {time.time() - ts}')


def _bench2(a, b):
    ts = time.time()
    r = isin_cffi(a, b)
    print(r.shape, r[:12])
    print(f'isin_cffi(): {time.time() - ts}')


if __name__ == "__main__":
    a = np.array(np.arange(10, 10000000, 2), dtype=np.int64)
    b = np.array(np.arange(10, 10000000, 3), dtype=np.int64)
    _bench(a, b)

    a = np.array(np.arange(10, 10000000, 2), dtype=np.int64)
    b = np.array(np.arange(10, 10000000, 3), dtype=np.int64)
    _bench2(a, b)
