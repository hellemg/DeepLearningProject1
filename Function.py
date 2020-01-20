import numpy as np


class Function:
    def __init__(self):
        self.name = 'hei'

    def do_the_math(sself, value):
        raise NotImplementedError

    def derivative(self, value):
        """
        Return derivate of the value
        """
        raise NotImplementedError


class ReLU(Function):
    def __init__(self):
        super().__init__()

    def derivative(self, value):
        if value <= 0:
            return 0
        else:
            return 1

    def apply_function(self, value):
        return np.maximum(0, value)


class Linear(Function):
    def __init__(self):
        super().__init__()

    def apply_function(self, value):
        return value

    def derivative(self, value):
        return 1


class TanH(Function):
    def __init__(self):
        super().__init__()

    def apply_function(self, value):
        return np.tanh(value)

    def derivative(self, value):
        return (np.cosh(value))**(-2)


class SoftMax(Function):
    def __init__(self):
        super().__init__()

    def apply_function(self, value):
        # Stable softmax to avoid NaN-problems
        exps = np.exp(value - np.max(value))
        return exps/np.sum(exps)

    def derivative(self, value):
        # https://deepnotes.io/softmax-crossentropy
        raise NotImplementedError


class L2:
    def __init__(self, activation):
        self.activation=activation

    def get_delta_w(self, label_value, estimated_value, x, z):
        return -1*np.sum((label_values - estimated_values)*x*self.activation.derivative(z))
