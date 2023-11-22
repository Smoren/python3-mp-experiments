import numpy as np
import numba as nb
import cffi
from llvmlite import ir
from numba.core.extending import intrinsic
from numba.core.typing import cffi_utils

import foo

ffi = cffi.FFI()
cffi_utils.register_module(foo)
foo_f = foo.lib.foo_f


@intrinsic
def ptr_to_val(typingctx, data):
    def impl(context, builder: ir.IRBuilder, signature, args):
        val = builder.load(args[0])
        return val

    # sig = nb.types.CPointer(nb.float64)(nb.float64)
    sig = data.dtype(nb.types.CPointer(data.dtype), 32)
    print(data.dtype)
    return sig, impl


@nb.njit
def test(a, b):
    a_wrap = np.int32(a)
    # This works for an array
    b_wrap = ffi.from_buffer(b.astype(np.float64))

    # np.frombuffer(ffi.buffer(b_wrap, 16 * 4), dtype=np.float64)
    # buf = ffi.buffer(b_wrap, 1)

    # np.frombuffer(ffi.buffer(b_wrap, b.shape[0] * ffi.sizeof('double')), dtype)
    foo_f(a_wrap, b_wrap)

    print(ptr_to_val(b_wrap))


a = 64.
b = np.ones(5)
test(a, b)
print(b)
