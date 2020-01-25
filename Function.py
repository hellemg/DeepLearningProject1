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
        return 1*(value>0)

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

    def nodes_error_in_layer(self, weights_transposed, error_in_next_layer, z):
        return weights_transposed*error_in_next_layer*self.activation.derivative(z)
