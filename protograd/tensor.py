import numpy as np

# wrapper around numpy arrays
class Tensor():
    def __init__(self, data, backward_path=[]):
        if isinstance(data, list):
            self.data = np.array(data).astype(np.float32)
        else: 
            self.data = data.astype(np.float32)

        self.remember_for_backward = backward_path

    def __str__(self):
        if len(self.remember_for_backward) == 0:
            return f"{self.data}"
        return f"{self.data}, \nbackward: {self.remember_for_backward[-1]}"

    def __int__(self):
        return Tensor(np.array(self.data, dtype=np.int32), backward_path=self.remember_for_backward)

    def __float__(self):
        return Tensor(np.array(self.data, dtype=np.float32), backward_path=self.remember_for_backward)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return Tensor(self.data[index], backward_path=self.remember_for_backward)

    def __array__(self, *type):
        return self.data
    
    def copy(self):
        return Tensor(self.data.copy(), backward_path=self.remember_for_backward)
    
    @property
    def T(self):
        return self.data.transpose()
    
    @property
    def dtype(self):
        return self.data.dtype
    
    @property
    def shape(self):
        return self.data.shape
    
    def reshape(self, shape):
        return Tensor(self.data.reshape(*(shape,)), backward_path=self.remember_for_backward)

    @staticmethod
    def zeros(*shape):
        return Tensor(np.zeros(shape, dtype=np.float32))

    @staticmethod
    def ones(*shape):
        return Tensor(np.ones(shape, dtype=np.float32))

    @staticmethod
    def randn(*shape):
        return Tensor(np.random.randn(*shape).astype(np.float32))

    @staticmethod
    def eye(dim):
        return Tensor(np.eye(dim).astype(np.float32))
    
    def to_numpy(self):
        return self.data.copy()

    def add_to_grad(self, layer):
        if isinstance(layer, type(object)):
            raise Exception("The input must be object.")
        elif layer in self.remember_for_backward:
            raise Exception(f"Already in backward memory. Initialize another {layer.__class__.__name__}")
        self.remember_for_backward.append(layer)
    
    def clear_grad(self):
        self.remember_for_backward = self.remember_for_backward.clear() # clear it for another loop

    def backward(self, *dinputs): # Backward and computing gardient
        for layer in self.remember_for_backward[-1::-1]:    # reverse list
            layer.backward(dinputs)
            dinputs = layer.dinputs
        self.dinputs = dinputs  # final derivative output

"""
class Function:
    def __call__(self, *inputs):
        return self.forward(*inputs)

class Add(Function):
    def forward(self, x, y):
        data = x + y
        return Tensor(data)
    
    def backward(self, dinputs):
        return dinputs
"""