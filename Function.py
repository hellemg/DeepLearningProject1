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

       
class L2:
    def __init__(self, activation):
        self.activation=activation

    def nodes_error_in_layer(self, weights_transposed, error_in_next_layer, z):
        return weights_transposed*error_in_next_layer*self.activation.derivative(z)

class CrossEntropy:
    def __init__(self, activation):
        self.activation=activation

    def gradient(y, s):
        """
        Return the gradient of cross-entropy of vectors y and s.

        :type y: ndarray
        :param y: one-hot vector encoding correct class

        :type s: ndarray
        :param s: softmax vector

        :returns: ndarray of size len(s)
        """
        return -y / s

class Activation:
    def __init__(self):
        pass

    def gradient(self, values):
        raise NotImplementedError

class SoftMax(Activation):
    def __init__(self):
        super().__init__()

    def softmax(x):
        """
        Return the Softmax of vector x, protected against under/overflow
        
        :type x: ndarray
        :param x: vector input
        
        :returns: ndarray of same length as x
        """
        x = x - np.max(x)
        exps = np.exp(x)
        return exps/np.sum(exps)

    def gradient(self, s):
        """
        Returns the Jacobian of the Softmax vector s

        :type s: ndarray
        :param s vector input

        :returns: ndarray of shape (len(s), len(s))
        """
        return np.diag(s) - np.outer(s,s)

    def jacobian(self, x):
        """
        Returns the jacobian of vector x, protected against under/overflow

        :type x: ndarray
        :param x: vector input
        
        :returns: ndarray of shape (len(x), len(x))
        """
        s = self.softmax(x)
        return self.gradient(s)