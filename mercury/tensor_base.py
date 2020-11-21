import numpy as np

class Tensor:
    def __init__(self, data, dtype=np.float32, grad=None, ctx=None):
        if isinstance(data, int) or isinstance(data, float):
            self.data = np.array([data])
        elif not isinstance(data, np.ndarray) and not isinstance(data, np.float32) \
            and not isinstance(data, Tensor) and not isinstance(data, list):
            raise Exception(f"Wrong input. Got {type(data)}; Should be list or array or number.")
        
        self.data = np.array(data, dtype=dtype)
        self.grad = grad   # gradient
        self._ctx = ctx    # own context (for function class)

    def __repr__(self):
        return f"Tensor({self.data}, backward={self._ctx})"
    
    def __len__(self):
        return len(self.data)
    
    def __abs__(self):
        return self.abs()
    
    def __pos__(self):
        return self.pos()
    
    def __neg__(self):
        return self.neg()
    
    def __add__(self, other):
        return self.add(other)
    
    def __sub__(self, other):
        return self.sub(other)
    
    def __mul__(self, other):
        return self.mul(other)
    
    def __div__(self, other):
        return self.div(other)
    
    def __truediv__(self, other):
        return self.div(other)
    
    def __pow__(self, other):
        return self.pow(other)
    
    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError()
        return self.data[index]
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def size(self):
        return self.data.size

    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def backward(self, allow_fill=True):
        # Chceck for state of tensor
        if self._ctx is None:
            return
        if self.grad is None and allow_fill:
            assert self.shape == (1,)  # fill in the first grad with ones
            self.grad = Tensor(np.ones_like(self.data), dtype=self.dtype)
        assert(self.grad is not None)

        topo = []   # topological order
        visited = set()
        def build_topo(x):
            visited.add(x)
            if x._ctx is not None:
                for child in x._ctx.parents:
                    if child not in visited:
                        build_topo(child)
                topo.append(x)
        build_topo(self)

        for tensor in reversed(topo):
            grads = tensor._ctx.backward(tensor._ctx, tensor.grad.data)
            grads = [grads] if len(tensor._ctx.parents) == 1 else grads
            for t, g in zip(tensor._ctx.parents, grads):
                if g is None:   continue
                #assert g.shape == t.shape, \
                #    f"Grad shape and Tensor shape don't match: {g.shape} != {t.shape} in {self._ctx}"
                t.grad = Tensor(g) if t.grad is None else (t.grad + Tensor(g))
