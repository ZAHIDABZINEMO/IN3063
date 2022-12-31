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
