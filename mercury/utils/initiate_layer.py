import numpy as np

def _uniform(a, b, dtype=np.float32):
    return np.random.uniform(-1., 1., size=(a, b)).astype(dtype) \
        / np.sqrt(a*b)

def _gaussian(a, b, dtype=np.float32):
    return np.random.randn(a, b).astype(dtype) \
        / np.sqrt(a * b)

def _xavier(a, b, dtype=np.float32):
    return np.random.uniform(-1., 1.).astype(dtype)\
        * np.sqrt(6./(a + b))

def _kaiming(a, b, dtype=np.float32):
    return np.random.randn(a, b).astype(dtype) \
        * np.sqrt(2./a*b)