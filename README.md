# Mercury
My own experimental neural network framework. In development...
Goal of the Mercury is to be mini-framework akin to pytorch, but I develop it mainly to be sure that I understand deep learning -> How to be sure? build it.

Even so it is aimed as mini-framework it should be sufficient for most of cases.

Should be alright with python 3.8.0+. Also it's builded on numpy and Tensor is just wrapper around it.
You can perform all basic numpy operation with Tensor. The tensor track and save all operation for
backpropagation (backward), so it would be used for computing gradient, which is used for updating weights and biases.

There is a lot of debugging to do and it still miss plenty of things, but you can already do something with it.

List to do:
- Upgrade Tensor
- add GPU acceleration (numba)
- implement basic NN

Inspired by: 
Pytorch:
https://github.com/pytorch/pytorch
https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html

Micrograd:
https://github.com/karpathy/micrograd


Also I found Tinygrad from George Hotz and learned a lot from it.
Tinygrad:
https://github.com/geohot/tinygrad

also this one -> MyGrad:
https://github.com/rsokl/MyGrad
