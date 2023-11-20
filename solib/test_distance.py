import math
import time
from ctypes import CDLL, c_double

distlib = CDLL('./distance.so')

distlib.l2norm.argtypes = [c_double, c_double]
distlib.l2norm.restype = c_double

distlib.l2distance.argtypes = [c_double, c_double, c_double, c_double]
distlib.l2distance.restype = c_double


def py_l2distance(x1, y1, x2, y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)


print(distlib.l2distance(1, 2, 3, 5))
print(py_l2distance(1, 2, 3, 5))

ts = time.time()
for _ in range(1000000):
    distlib.l2distance(1, 2, 3, 5)
print(time.time() - ts)

ts = time.time()
for _ in range(1000000):
    py_l2distance(1, 2, 3, 5)
print(time.time() - ts)
