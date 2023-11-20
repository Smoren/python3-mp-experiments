from ctypes import CDLL, c_double

distlib = CDLL('./distance.so')

distlib.l2norm.argtypes = [c_double, c_double]
distlib.l2norm.restype = c_double

distlib.l2distance.argtypes = [c_double, c_double, c_double, c_double]
distlib.l2distance.restype = c_double

print(distlib.l2distance(1, 2, 3, 5))
print(distlib.l2norm(500, 200))
