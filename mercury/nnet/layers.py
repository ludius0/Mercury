import numpy as np

from mercury.operation_base import *
from mercury.tensor_base import Tensor
from mercury.utils.initiate_layer import _uniform


class Layer():
    def __call__(self, *inputs):
        for t in inputs:
            if not isinstance(t, Tensor):
                raise TypeError(f"Input should be Tensor. Got {type(t)}.")

class Linear(Layer):
    def __init__(self, n_inputs, n_neurons, biases=False, init_func="_uniform()", dtype=np.float32):
        self.weights = Tensor(_uniform(n_inputs, n_neurons, dtype=dtype))
        #self.biases = Tensor(np.zeros((1, n_neurons)), dtype=np.int32) #if not biases else Tensor(_uniform(1, n_neurons, dtype=dtype))

    def __call__(self, input):
        super(Linear, self).__call__(input)
        output = input.dot(self.weights) #+ self.biases
        return output

class Connv2d(Layer):
    pass