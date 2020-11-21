import numpy as np

from mercury.operation_base import *

class Reshape(Function):
    @staticmethod
    def forward(ctx, input, *shape):
        ctx.save_for_derivative(input.shape)
        return input.reshape(shape)
    
    @staticmethod
    def backward(ctx, grad_output):
        input_shape,  = ctx.saved_tensor
        return grad_output.reshape(input_shape)
setattr_tensor(Reshape)

class Transpose(Function):
    @staticmethod
    def forward(ctx, input):
        return input.T
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.T
setattr_tensor(Transpose)

class GetItem(Function):
    pass

class Flatten(Function):
    pass

class Astype(Function):
    @staticmethod
    def forward(ctx, input, type):
        if type not in _dtypes:
            raise TypeError()
        ctx.save_for_derivative(input.dtype)
        return input.astype(type)
    
    @staticmethod
    def backward(ctx, grad_output):
        type = ctx.saved_tensors
        return grad_output.astype(type)
setattr_tensor(Astype)