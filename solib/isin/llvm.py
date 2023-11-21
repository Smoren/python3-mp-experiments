from ctypes import CFUNCTYPE, c_int64, c_bool, POINTER

import numpy as np
from numpy.ctypeslib import ndpointer

import llvmlite.binding as llvm
import llvmlite.ir as ir
import numba

llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

llvm.load_library_permanently('./isin.so')

int_type = ir.IntType(64)
int_ptr_type = ir.PointerType(int_type)
fnty = ir.FunctionType(int_ptr_type, (int_ptr_type, int_type, int_ptr_type, int_type))

module = ir.Module("myisin")

# llvm IR call external C++ function
outfunc = ir.Function(module, fnty, name="isin")
# just declare shared library function in module
outaddfunc = ir.Function(module, fnty, name="export_isin")
builder = ir.IRBuilder(outaddfunc.append_basic_block(name="entry"))
outresult = builder.call(outfunc, outaddfunc.args)
builder.ret(outresult)

strmod = str(module)
assmod = llvm.parse_assembly(strmod)
assmod.verify()

target = llvm.Target.from_default_triple()
target_machine = target.create_target_machine()
engine = llvm.create_mcjit_compiler(assmod, target_machine)
engine.finalize_object()

func_ptr = engine.get_function_address("export_isin")

isin = CFUNCTYPE(
    POINTER(c_bool),
    POINTER(c_int64),
    c_int64,
    POINTER(c_int64),
    c_int64,
)(func_ptr)


@numba.njit(fastmath=True, parallel=True)
def test_myadd():
    for _ in numba.prange(10):
        a, b = 2, 4
        res = isin([0], 1, [1], 1)
        print(res)


test_myadd()
