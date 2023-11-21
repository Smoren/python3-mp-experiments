import numba as nb
import numpy as np
from cffi import FFI
from numba.core import cgutils
from numba.core.extending import intrinsic

ffi = FFI()
ffi.cdef('bool* isin(int64_t where[], int64_t where_size, int64_t what[], int64_t what_size);')

# loads the entire C namespace
C = ffi.dlopen('./isin.so')
c_isin = C.isin


@intrinsic
def val_to_ptr(typingctx, data):
    def impl(context, builder, signature, args):
        ptr = cgutils.alloca_once_value(builder, args[0])
        return ptr
    sig = nb.types.CPointer(nb.typeof(data).instance_type)(nb.typeof(data).instance_type)
    return sig, impl



@intrinsic
def ptr_to_val(typingctx, data):
    def impl(context, builder, signature, args):
        val = builder.load(args[0])
        return val
    sig = data.dtype(nb.types.CPointer(data.dtype))
    return sig, impl


@intrinsic
def get_ptr(typingctx, data):
    def impl(context, builder, signature, args):
        ptr = cgutils.alloca_once_value(builder, args[0])
        return ptr
    sig = nb.types.CPointer(nb.int64[:])(nb.int64[:])
    return sig, impl


@nb.njit
def cffi_isin_example(x):
    a = get_ptr(np.array([11, 2, 3], dtype=np.int64))
    b = get_ptr(np.array([22, 2, 3], dtype=np.int64))
    print(ptr_to_val(a), ptr_to_val(b))
    # return c_isin(a, 3, b, 3)


print(cffi_isin_example(10))

#
# @nb.njit
# def print_ptr():
#     a = np.array([1, 2, 3], dtype=np.int64)
#     address = a.ctypes.data
#     val_to_ptr(a)
#
#     # print(address)
#     # pointer, read_only_flag = a.__array_interface__['data']
#     # print(pointer)
#
# print_ptr()
