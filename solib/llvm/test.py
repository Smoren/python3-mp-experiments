from ctypes import CFUNCTYPE, c_double

import llvmlite.binding as llvm
import llvmlite.ir as ir
import numba

llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

llvm.load_library_permanently('./fdadd.so')

double = ir.DoubleType()
fnty = ir.FunctionType(double, (double, double))

module = ir.Module("fdadd")
func = ir.Function(module, fnty, name="fpadd")

block = func.append_basic_block(name="entry")
builder = ir.IRBuilder(block)
a, b = func.args
result = builder.fadd(a, b, name="res")
builder.ret(result)

strmod = str(module)
assmod = llvm.parse_assembly(strmod)
assmod.verify()

target = llvm.Target.from_default_triple()
target_machine = target.create_target_machine()
engine = llvm.create_mcjit_compiler(assmod, target_machine)
engine.finalize_object()

func_ptr = engine.get_function_address("fpadd")

myadd = CFUNCTYPE(c_double, c_double, c_double)(func_ptr)


@numba.njit(fastmath=True, parallel=True)
def test_myadd():
    for _ in numba.prange(10):
        res = myadd(1.0, 3.5)
        print("myadd(...) =", res)


test_myadd()
