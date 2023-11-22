import numpy as np
import ctypes
from numba import jit, njit

lib = ctypes.cdll.LoadLibrary('./simpleLoopC.so')
loop=lib.loopTestC

loop.restype = None

loop.argtypes = [ctypes.c_size_t,
        np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]


# @njit(parallel=False)
def testC():
    a = np.arange(100, dtype=np.float64)
    b = np.zeros_like(a)

    print(a, b)
    n = a.size
    loop(n, a, b)
    print(a, b)


testC()
