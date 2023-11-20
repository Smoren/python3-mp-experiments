import numba as nb

@nb.cfunc("float64(float64, float64)")
def add(x, y):
    return x + y

print(add.ctypes(4.0, 5.0))  # prints "9.0"
