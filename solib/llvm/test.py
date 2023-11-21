import ctypes
from ctypes import CFUNCTYPE, c_double
from numpy.ctypeslib import ndpointer

import llvmlite.binding as llvm
import llvmlite.ir as ir
import numba
import numpy as np

llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

llvm.load_library_permanently('./fdadd.so')

double = ir.DoubleType()
fnty = ir.FunctionType(double, (double, double))

module = ir.Module("fdadd")

# llvm IR call external C++ function
outfunc = ir.Function(module, fnty, name="fpadd")
# just declare shared library function in module
outaddfunc = ir.Function(module, fnty, name="outadd")
builder = ir.IRBuilder(outaddfunc.append_basic_block(name="entry"))
a, b = outaddfunc.args
outresult = builder.call(outfunc, (a, b))
builder.ret(outresult)

strmod = str(module)
assmod = llvm.parse_assembly(strmod)
assmod.verify()

target = llvm.Target.from_default_triple()
target_machine = target.create_target_machine()
engine = llvm.create_mcjit_compiler(assmod, target_machine)
engine.finalize_object()

func_ptr = engine.get_function_address("outadd")

myisin = CFUNCTYPE(
    numba.types.CPointer(ctypes.c_int64),
    ndpointer(ctypes.c_int64, flags="C_CONTIGUOUS"),
    ctypes.c_int64,
    ndpointer(ctypes.c_int64, flags="C_CONTIGUOUS"),
    ctypes.c_int64,
)(func_ptr)


@numba.njit(fastmath=True)
def test_myadd():
    numba.types.CPointer(ctypes.c_int64)
    a = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    b = np.array([1, 3, 5], dtype=np.int64)

    res = myisin(ctypes.c_void_p(a.ctypes.data), ctypes.c_int64(a.shape[0]), ctypes.c_void_p(b.ctypes.data), ctypes.c_int64(b.shape[0]))
    print("myadd(...) =", res)


test_myadd()
