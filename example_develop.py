import numpy as np
from mercury import Tensor
from mercury.nnet.layers import Linear

x = Tensor([-1., 7.9, 4.3])
y = Tensor([2.1, 5.8, 9.2])
z = Tensor([8.3, 5.1, -1.2])

y = y / z
q = x.mul(y).relu().sum()
q.backward()

print("-"*20)
print(x.grad)
print(y.grad)
print(z.grad)



### NOTES:
# Check if all functions and classes and Tensor's operators/instances works (checkout Dot)
# Implement Linear layer + edit files (add random folder ans so on...)
# Implement Sigmoid
# Implement activation functions to be callable as function Sigmoid(Tensor)
# Implement Optimizer
# Write an example with mnist

# additional:
# make universal settings for dtype
# implement loss functions
# implement connv1d and 2d
# implement rnn