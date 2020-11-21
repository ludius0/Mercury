import numpy as np

from mercury.tensor_base import Tensor

def empty(*shape, dtype=np.float32):
    return Tensor(np.empty(shape, dtype=dtype))

def zeros(*shape, dtype=np.float32):
    return Tensor(np.zeros(shape, dtype=dtype))

def ones(*shape, dtype=np.float32):
    return Tensor(np.ones(shape, dtype=dtype))

def eye(*shape, dtype=np.float32):
    return Tensor(np.eye(shape).astype(dtype))

### RANDOMNESS ###

# Note: implement this as Tensor.random.radn() or mercury.random.radn() (probably this one)

def randn(*shape, dtype=np.float32):
    return Tensor(np.random.randn(shape).astype(dtype))

def rand(*shape, dtype=np.float32):
    return Tensor(np.random.rand(shape).astype(dtype))

def sample(*shape, dtype=np.float32):
    return Tensor(np.random.sample(shape).astype(dtype))

def seed(seed, dtype=np.float32):
    """
    Set seed for randomness with numpy.
    """
    return np.random.seed(seed)