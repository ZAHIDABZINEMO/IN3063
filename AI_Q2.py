import numpy as np

class Sigmoid:
    def forward(self, x):
        """
        Applies the sigmoid function element-wise to the input tensor x.
        """
        return 1 / (1 + np.exp(-x))

    def backward(self, x, grad_output):
        """
        Computes the gradient of the sigmoid function with respect to the input tensor x.
        """
        # Compute the sigmoid function
        sigmoid = 1 / (1 + np.exp(-x))
        # Calculate the gradient
        grad_input = grad_output * sigmoid * (1 - sigmoid)
        return grad_input

class ReLU:
    def forward(self, x):
        """
        Applies the ReLU function element-wise to the input tensor x.
        """
        return np.maximum(0, x)

    def backward(self, x, grad_output):
        """
        Computes the gradient of the ReLU function with respect to the input tensor x.
        """
        grad_input = grad_output
        grad_input[x <= 0] = 0
        return grad_input

class Softmax:
    def forward(self, x):
        """
        Applies the softmax function element-wise to the input tensor x.
        """
        # Subtract max(x) element-wise to avoid numerical instability
        x = x - np.max(x, axis=1, keepdims=True)
        # Apply softmax function element-wise to x
        exp = np.exp(x)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def backward(self, x, grad_output):
        """
        Computes the gradient of the softmax function with respect to the input tensor x.
        """
        # Calculate the output of the forward pass
        output = self.forward(x)
        # Calculate the gradient
        return output * (grad_output - np.sum(grad_output * output, axis=1, keepdims=True))
class Dropout:
    def __init__(self, p=0.5):
        """
        Initializes the dropout layer with a dropout probability p.
        """
        self.p = p
        self.mask = None

    def forward(self, x, train=True):
        """
        Applies dropout to the input tensor x during training.
        """
        if train:
            # Generate a mask with probability p of keeping each element
            self.mask = np.random.rand(*x.shape) > self.p
            # Apply the mask to the input tensor
            return x * self.mask
        else:
            # During evaluation, simply return the input tensor
            return x

    def backward(self, grad_output):
        """
        Propagates the gradient of the loss through the dropout layer.
        """
        # Apply the mask to the gradient
        return grad_output * self.mask
