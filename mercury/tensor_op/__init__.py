from ._ops import *
from ._funcs import *

functions = [empty, zeros, ones, eye, randn, rand, sample, seed]
for func in functions:
    setattr(Tensor, func.__name__, func)