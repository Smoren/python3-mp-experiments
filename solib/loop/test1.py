import numpy as np
import numba as nb
import ctypes
from numba import jit, njit
from numba.core.extending import intrinsic


@intrinsic
def the_loop(typingctx, *args):
    # lib = ctypes.cdll.LoadLibrary('./simpleLoopC.so')
    # loop = lib.loopTestC
    # loop.restype = None
    # loop.argtypes = [
    #     ctypes.c_int64,
    #     np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),
    #     np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),
    # ]

    def impl(context, builder, signature, args):
        # loop(args[0], args[1], arg[2])
        return None

    sig = nb.types.NoneType('none')()
    return sig, impl





@njit(parallel=False)
def testC():
    a = np.arange(100, dtype=np.float64)
    b = np.zeros_like(a)

    print(a, b)
    n = a.size
    the_loop(n, a, b)
    print(a, b)


testC()
