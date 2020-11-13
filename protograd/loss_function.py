import numpy as np
from nn import Softmax

class Loss:
    def __call__(self, inputs, y):
        return self.calculate(inputs, y)

    def calculate(self, inputs, y):
        inputs = inputs.data
        y = np.array(y.data, dtype=np.uint8)
        sample_losses = self.forward(inputs, y)
        loss = np.mean(sample_losses)   # erros or loss
        self.backward(loss, y)
        return loss


class CrossEntropyLoss(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)   # n of samples in a batch
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7) # clipping from preventing division by 0 + don't dragging mean towards any value

        # probabilities for target values
        if len(y_true.shape) == 1:      # vector
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # Mask values -> for one-hot encoded labels
        elif len(y_true.shape) == 2:    # matrix
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        
        return -np.log(correct_confidences) # negative log likelihoods
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:  # if labels are sparse, than turn them into one-hot vector
            y_true = np.eye(labels)[y_true]
        
        self.dinputs = (-y_true / dvalues) / samples # calculate and normalize gradient


class SoftmaxCrossEntropyLoss:
    def __init__(self):
        """
        Combining Softmax and CrossEntropyLoss
        for better speed (7x),  
        when calculating the gradient.
        """
        self.activation = Softmax()
        self.loss = CrossEntropyLoss()
    
    def __call__(self, inputs, target):
        self.y_true = np.array(target.data, dtype=np.uint8)
        inputs.add_to_grad(self)
        self.forward(inputs, self.y_true)
        self.dvalues = self.output
        return self.loss_output

    def forward(self, prediction, y_true):
        self.activation.forward(prediction)
        self.output = self.activation.output
        self.loss_output = np.mean(self.loss.forward(self.output, self.y_true))
    
    def backward(self, dvalues):
        samples = len(self.dvalues)
        if len(self.y_true.shape) == 2:
            self.y_true = np.argmax(self.y_true, axis=1)
        self.dinputs = self.dvalues.copy()
        self.dinputs[range(samples), self.y_true] -= 1
        self.dinputs = self.dinputs / samples


class MSELoss:
    def __init__(self):
        pass