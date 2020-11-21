import numpy as np

from .tensor_base import Tensor

class Function:
    def __init__(self, *tensors, **kwargs):
        self.parents = tensors
        self.saved_tensors = []     # rename this on saved_values
    
    def save_for_derivative(self, *tensor): # rename this on save_for_backward(self, *tensors)
        self.saved_tensors.extend(tensor)

    def apply(self, *args, **kwargs):
        ctx = self(*args)   # context
        ret = Tensor(self.forward(ctx, *[t.data for t in args], **kwargs))  # input data (np.array) from Tensor
        ret._ctx = ctx
        print(ret)
        return ret

def setattr_tensor(cls):
    # cls -> operation (op)
    def call_func(*args, **kwargs):
        # check if all *args are tensors
        remember = []
        for t in args:
            if not isinstance(t, Tensor):
                t = Tensor(t)   # if not Tensor; convert to Tensor
            remember.append(t)
        args = tuple(i for i in remember)
        return cls.apply(cls, *[t for t in args], **kwargs)
    setattr(Tensor, cls.__name__.lower(), call_func)

# Supported dtypes
_dtypes = [np.int16, np.int32, np.int64,
        np.float16, np.float32, np.float64]