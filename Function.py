import numpy as np


class Activation:
    def __init__(self):
        pass

    def apply_function(self):
        raise NotImplementedError

    def gradient(self, values):
        raise NotImplementedError


class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def apply_function(self, z):
        """
        Return the ReLU of vector z

        :type z: ndarray
        :param z: vector input

        :returns: ndarray of same length as z        
        """
        return np.maximum(0, z)

    def gradient(self, z):
        """
        Return the gradient of ReLU with respect to vector z

        :type z: ndarray
        :param z: vector input

        :returns: ndarray of same length as z        
        """
        return 1*(z > 0)


class Linear(Activation):
    def __init__(self):
        super().__init__()

    def apply_function(self, z):
        """
        Return the Linear of vector z

        :type z: ndarray
        :param z: vector input

        :returns: ndarray of same length as z
        """
        return z

    def gradient(self, z):
        """
        Return the gradient of Linear with respect to vector z

        :type z: ndarray
        :param z: vector input

        :returns: ndarray of same length as z        
        """
        return np.ones_like(z)


class TanH(Activation):
    def __init__(self):
        super().__init__()

    def apply_function(self, z):
        """
        Return the TanH of vector z

        :type z: ndarray
        :param z: vector input

        :returns: ndarray of same length as z
        """
        return np.tanh(z)

    def gradient(self, z):
        """
        Return the gradient of TanH with respect to vector z

        :type z: ndarray
        :param z: vector input

        :returns: ndarray of same length as z        
        """
        return (np.cosh(z))**(-2)

    def __init__(self):
        super().__init__()

    def apply_function(z):
        """
        Return the Softmax of vector z, protected against under/overflow

        :type z: ndarray
        :param z: vector input

        :returns: ndarray of same length as z
        """
        z = z - np.max(z)
        exps = np.exp(z)
        return exps/np.sum(exps)

    def gradient(self, s):
        """
        Returns the Jacobian of the Softmax vector s

        :type s: ndarray
        :param s vector input

        :returns: ndarray of shape (len(s), len(s))
        """
        return np.diag(s) - np.outer(s, s)

    def jacobian(self, z):
        """
        Returns the jacobian of vector z, protected against under/overflow

        :type z: ndarray
        :param z: vector input

        :returns: ndarray of shape (len(z), len(z))
        """
        s = self.softmax(z)
        return self.gradient(s)
