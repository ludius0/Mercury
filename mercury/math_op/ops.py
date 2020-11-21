import numpy as np

from mercury.operation_base import *

class Abs(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_derivative(x)
        return np.absolute(x)

    @staticmethod
    def backward(ctx, grad_output):
        return np.absolute(grad_output)
setattr_tensor(Abs)

class Pos(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_derivative(x)
        return np.positive(x)

    @staticmethod
    def backward(ctx, grad_output):
        return np.positive(grad_output)
setattr_tensor(Pos)

class Neg(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_derivative(x)
        return np.negative(x)

    @staticmethod
    def backward(ctx, grad_output):
        return np.negative(grad_output)
setattr_tensor(Neg)

class Add(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_derivative(x.shape, y.shape)
        return x + y

    @staticmethod
    def backward(ctx, grad_output):
        shape_x, shape_y = ctx.saved_tensors
        return grad_output, grad_output
setattr_tensor(Add)

class Sub(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_derivative(x.shape, y.shape)
        return x - y

    @staticmethod
    def backward(ctx, grad_output):
        shape_x, shape_y = ctx.saved_tensors
        return grad_output, -grad_output
setattr_tensor(Sub)

class Mul(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_derivative(x, y)
        return x * y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return y * grad_output, x * grad_output
setattr_tensor(Mul)

class Div(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_derivative(x, y)
        return x / y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return grad_output / y, -x * grad_output / y**2
setattr_tensor(Div)

class Pow(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_derivative(x, y)
        return x ** y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        inp1, inp2 = y * (x**(y - 1.0)) * grad_output, (x**y) * np.log(x) * grad_output
        return inp1, inp2
setattr_tensor(Pow)

class Dot(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_derivative(x, y)
        return x.dot(y)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_output.dot(y.T)
        grad_y = x.T.dot(grad_output)
        return grad_x, grad_y
setattr_tensor(Dot)

class Sum(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_derivative(x)
        return np.array([x.sum()])

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * np.ones_like(input)
setattr_tensor(Sum)

class Invert(Function):
    @staticmethod
    def forward(ctx, x):
        return np.invert(x)

    @staticmethod
    def backward(ctx, grad_output):
        return np.invert(grad_output)
setattr_tensor(Invert)