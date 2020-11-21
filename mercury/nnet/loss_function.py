import numpy as np

from mercury.tensor_base import Tensor

class Loss():
    def __call__(self, *inputs):
        if not isinstance(*[t for t in inputs], Tensor):
            raise TypeError(f"Input should be Tensor. Got {type(t)}.")

class MSELoss(Loss):
    def __call__(self, input):
        super(SGD, self).__call__(input)
        pass