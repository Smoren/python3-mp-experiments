import ctypes

import numpy as np
from llvmlite import ir
import numba as nb
from numba import types, njit
from numba.core import cgutils
from numba.core.extending import intrinsic


@intrinsic
def get_ptr(typingctx, data, data1):
    lib = ctypes.cdll.LoadLibrary('./simpleLoopC.so')
    loop = lib.loopTestC
    loop.restype = None
    loop.argtypes = [
        ctypes.c_int64,
        np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),
    ]

    def impl(context, builder, signature, args):
        print(args[0])
        ptr = cgutils.alloca_once_value(builder, args[0])
        return ptr
    sig = nb.types.CPointer(nb.int64[:])(nb.int64)
    return sig, impl


@njit
def mul(n, a, b):
    get_ptr(a, b)


n = 3
a = np.array([1, 2, 3])
b = np.array([0, 0, 0])
print(mul(n, a, b))
