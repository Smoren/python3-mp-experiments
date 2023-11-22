import cffi

ffi = cffi.FFI()
defs = "void foo_f(double a, double *b);"
ffi.cdef(defs, override=True)
source = """
void foo_f(double a, double *b) { b[2]+=a; printf("[%f]", b[2]); }
"""
ffi.set_source(module_name="foo", source=source)
ffi.compile()


