import numpy as np
from tensor import Tensor

class Module:
    def __call__(self, inputs):
        inputs.add_to_grad(self)
        self.forward(inputs)
        return Tensor(self.output, backward_path=inputs.remember_for_backward)


class Sequential(Module):
    def __init__(self, *sequence):
        self.forward_sequence = list(*sequence)
        self.optim_layers = ["LinearLayer"]
    
    def __call__(self, inputs):
        for layer in self.forward_sequence:
            inputs = layer(inputs)

    def append(self, layer):
        self.forward_sequence.append(layer)
    
    def forward(self, inputs):
        for layer in self.forward_sequence:
            inputs = layer.forward(inputs)
        return inputs


class LinearLayer(Module):
    def __init__(self, n_inputs, n_neurons):
        #self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.weights = (np.random.uniform(-1., 1., size=(n_inputs, n_neurons)) \
            / np.sqrt(n_inputs*n_neurons)).astype(np.float32)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

""" ------ ACTIVATION FUNCTIONS ------ """

class ReLU(Module):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= np.array([0])] = 0


class Sigmoid(Module):
    def forward(self, inputs):
        self.output = 1 / 1 + np.exp(1)**-inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs = (1 / 1 + np.exp(1)**-self.dinputs) / (1 - 1 / 1 + np.exp(1)**(-self.dinputs))


class Softmax(Module):
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))     # exponencial values
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)    # probabilities
    
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diaflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)