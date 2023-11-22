import unittest

from numba import njit
from numba.tests.support import MemoryLeakMixin, TestCase
from numba.extending import intrinsic
from numba.core import typing, types


@intrinsic
def foo(typingctx):
    dtype = types.int64
    rettype = dtype
    sig = rettype()

    def make_list():
        out = []
        out.append(1)
        out.append(2)
        return out

    def get_list_element(a, i):
        return a[i]

    def codegen(context, builder, signature, args):
        # Generate a temporary list
        inner_argtypes = []
        inner_rettype = types.List(dtype)
        inner_sig = typing.signature(inner_rettype, *inner_argtypes)
        inner_args = []
        tmp_list = context.compile_internal(
            builder, make_list, inner_sig, inner_args)

        # get a single element from the list
        list_idx = context.get_constant(types.intp, 0)
        data = context.compile_internal(builder, get_list_element,
                                        dtype(inner_rettype, types.intp),
                                        [tmp_list, list_idx])

        # decref anything that can possibly be freed
        if context.enable_nrt:
            context.nrt.decref(builder, inner_rettype, tmp_list)
            context.nrt.decref(builder, dtype, data)

        print(data)
        return data

    return sig, codegen


@njit
def bar():
    return foo()


results = bar()
print(results)
