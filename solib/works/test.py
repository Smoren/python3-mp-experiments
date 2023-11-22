import numpy as np
import numba as nb
import cffi
from numba.core.typing import cffi_utils

import foo

ffi = cffi.FFI()
cffi_utils.register_module(foo)
foo_f = foo.lib.foo_f


@nb.njit
def test(a, b):
    a_wrap = np.int32(a)
    b_wrap = ffi.from_buffer(b.astype(np.float64))
    foo_f(a_wrap, b_wrap)
    return nb.carray(b_wrap, b.shape[0], b.dtype)


a = 64.
b = np.ones(5)
r = test(a, b)
print(r)
