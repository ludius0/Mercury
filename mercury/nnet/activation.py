import numpy as np

from mercury.operation_base import *

class ReLU(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_derivative(input)
        return np.maximum(input, 0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.copy()
        grad_input[input < 0] = 0
        return grad_input
setattr_tensor(ReLU)

class LeakyReLU(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        pass

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        pass
setattr_tensor(LeakyReLU)

class Sigmoid(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        pass

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        pass
setattr_tensor(Sigmoid)

class Tanh(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        pass

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        pass
setattr_tensor(Tanh)