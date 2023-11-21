---
author: Pearu Peterson
created: 2021-10-05
---

> A revised version of this blog post is available as [Quansight Labs blog](https://labs.quansight.org/blog/2021/10/cxx-numba-interoperability/).

# An efficient method of calling C++ functions from numba/Python using clang++/ctypes/rbc

## Introduction

The aim of this document is to explore a method of calling C++ library
functions from [Numba](http://numba.pydata.org/) [jitted
functions](https://numba.pydata.org/numba-doc/latest/user/jit.html#compiling-python-code-with-jit)
--- Python functions that are decorated with
``numpa.jit(nopython=True)``.

While there exists ways to wrap C++ codes to Python, calling these
wrappers from Numba jittied functions may not be as straightforward
and efficient as one would hope. The underlying problem to inefficency
is that the Python/C++ wrappers are designed to be called on Python
objects. For instance, with keeping in mind the aim of this post, a
call to a Numba jitted function would involve converting a Python
object to a low-level object, say, to some C/C++ equivalent intrinsic
type or structure, then converting it back to Python object to be
passed to the Python/C++ wrapper function, that would convert the
Python object (again) to an equivalent C/C++ type object which can be
finally passed to the underlying C++ library function. The return
value of the C++ function would be transformed several times as well,
just in the opposite direction of the functions calling sequence. In
sum, all these object transformations may build up a considerable
overhead of calling otherwise highly efficient C++ library functions
from Python.

Another difficulty with calling C++ library functions from Numba
originates from
[name-mangling](https://en.wikipedia.org/wiki/Name_mangling) that C++
compilers apply to function names in order to support function/method
overloadings as well as other relevant C++ language features. In
principle, one would be able to call such a C++ library function
directly from Numba jitted function (using
[numba.cfunc](https://numba.pydata.org/numba-doc/latest/user/cfunc.html)
feature) if one would know how the C++ compiler transforms the
function name internally . However, the name-mangling algorithm is C++
compiler vendor dependent, and in practice, it would be hard to
predict the mangled name in a portable manner.

## cxx2py.py tool

In this post, we'll introduce a straightforward and efficient method
of calling C++ library functions from Numba jitted functions that
circumvents the name-mangling problem. The method is based on
determining the C++ library functions addresses at runtime that
together with functions signatures can be used to set up an highly
efficient calling sequence.  This method will require creating a small
C/C++ wrapper library that contains ``export "C"``-attributed
functions which return the addresses of C++ libray functions at
runtime.

A Python script [cxx2py.py](cxx2py/cxx2py.py) is provided that
auto-generates, from user-supplied C++ header and source files, the
wrapper library as well as a Python
[ctypes](https://docs.python.org/3/library/ctypes.html) wrapper
module. The Python module contains ``ctypes`` definitions of C++
library functions that Numba jitted functions are able to call
directly without requiring the expensive object transformations
mentioned above.

Currently, the supported features include:

- wrapping of C++ library functions with scalar inputs and return
  values,
- supported C++ functions may be defined inside C++ namespaces,
- supported C++ functions may be static class member functions.

The ``cxx2py.py`` tool can be extended to support other C++ features
such as

- creating C++ class/struct instances from Python
- passing C++ class/struct instances around
- calling the methods of C++ class/struct instances
- supporting pointer types as function inputs and return values
- etc.

``cxx2py.py`` uses ``clang++`` to parse C++ header files and to build
the shared wrapper library and the Python wrapper module. It also uses
[RBC - Remote Backend Compiler](https://github.com/xnd-project/rbc/)
for parsing the signatures of C++ functions for convenience.

## Example

As a prerequisite, let's create a conda environment as follows
```bash
$ conda create -n cxx2py-demo -c conda-forge numba rbc cxx-compiler clangdev
$ conda activate cxx2py-demo
```

We assume that the [cxx2py.py](cxx2py/cxx2py.py) script is copied to
the current working directory and is functional:
```bash
$ python cxx2py.py --help
usage: cxx2py.py [-h] [-m MODULENAME] [--clang-exe CLANG_EXE]
                 [--clang-ast-dump-flags CLANG_AST_DUMP_FLAGS]
                 [--clang-build-flags CLANG_BUILD_FLAGS]
                 [--clang-extra-flags CLANG_EXTRA_FLAGS]
                 [--build] [--verbose]
                 file [file ...]

Generate ctypes wrappers to C++ library functions

positional arguments:
  file                  C++ header/source file

optional arguments:
  -h, --help            show this help message and exit
  -m MODULENAME, --modulename MODULENAME
                        Python module name of ctypes wrappers (default: untitled)
  --clang-exe CLANG_EXE
                        Path to clang compiler (default: clang++)
  --clang-ast-dump-flags CLANG_AST_DUMP_FLAGS
                        Override flags to clang ast dump command (default:
                        '-Xclang -ast-dump -fsyntax-only -fno-diagnostics-color')
  --clang-build-flags CLANG_BUILD_FLAGS
                        Override flags to clang build shared library command
                        (default: '-shared -fPIC')
  --clang-extra-flags CLANG_EXTRA_FLAGS
                        Extra flags to clang command (default: '')
  --build               Build shared library (default: False)
  --verbose             Be verbose (default: False)
```

Next, let us consider the following C++ header and source file that we
will use as a model of a C++ library:
```c++
/* File: foo.hpp */
#include <iostream>
int foo(int a);

namespace ns {
  namespace ns2 {
    double bar(double x);
    class BarCls {
    public:
      BarCls(double a): a_(a) {}
      double get_a() { return a_; }
      static int fun() { return 54321; }
    private:
      double a_;
    };
  }
}

/* File: foo.cpp */
int foo(int a) {
  std::cout << "in foo(" << std::to_string(a) << ")" << std::endl;
  return a + 123;
}

namespace ns { namespace ns2 {
    double bar(double a) {
      std::cout << "in ns::ns2::bar(" << std::to_string(a) << ")" << std::endl;
      return a + 12.3;
    }
}}
```

We build a ctypes wrapper library using
```bash
$ python cxx2py.py -m libfoo foo.hpp foo.cpp --build
DONE

As a quick test, try running:

  LD_LIBRARY_PATH=. python -c "import untitled as m; print(m.__all__)"
```
that will create three files into the current directory:
```bash
$ ls *libfoo*
cxx2py_libfoo.cpp  libcxx2py_libfoo.so  libfoo.py
```

Notice that the generated
[cxx2py_libfoo.cpp](cxx2py/cxx2py_libfoo.cpp) file contains
light-weight C functions for returning the addresses of C++ functions:
```c++
#include <memory>
#include <cstdint>
#include "foo.hpp"

extern "C" intptr_t get_foo_address() {
  /* int (int) */
  return reinterpret_cast<intptr_t>(std::addressof(foo));
}


extern "C" intptr_t get_ns__ns2__bar_address() {
  /* double (double) */
  return reinterpret_cast<intptr_t>(std::addressof(ns::ns2::bar));
}


extern "C" intptr_t get_ns__BarCls__fun_address() {
  /* int () */
  return reinterpret_cast<intptr_t>(std::addressof(ns::BarCls::fun));
}
```

The ``cxx2py_libfoo.cpp`` file is built into the shared library
``libcxx2py_libfoo.so`` when ``--build`` flag is used.

Let's test the wrapper module [libfoo](cxx2py/libfoo.py) in Python:
```bash
$ export LD_LIBRARY_PATH=.  # this makes sure that ctypes is able to find the shared library
```
```python
$ python
>>> import libfoo
>>> libfoo.__all__
['foo', 'ns__ns2__bar', 'ns__BarCls__fun']
>>> libfoo.foo(5)
in foo(5)
128
>>> libfoo.ns__ns2__bar(1.2)
in ns::ns2::bar(1.200000)
13.5
>>> libfoo.ns__BarCls__fun()
54321
```
that is, the C++ library functions can be called directly from Python
thanks to [ctypes](https://docs.python.org/3/library/ctypes.html)!.

Moreover, the C++ library functions can be called from Numba jitted
functions as well. For example:
```python
>>> import numba
>>> @numba.njit
... def fun(x):
...     return libfoo.foo(x + 2)
... 
>>> fun(5)
in foo(7)
130
```

## Summary

In this post, we outlined a method of calling C++ library functions
from Python with emphasis of using these from Numba jitted functions
with minimal overhead. While the provided tool
[cxx2py.py](cxx2py/cxx2py.py) currently supports only
wrapping C++ functions with scalar inputs and return values, it can be
easily extended to support other C++ features as well.
