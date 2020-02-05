import numpy as np


class Activation:
    def __init__(self):
        pass

    def apply_function(self):
        raise NotImplementedError

    def gradient(self, values):
        raise NotImplementedError

    def backpropagate(self, next_der):
        # backpropagate partial derivative from next layer 
        """
        :type next_der: ndarray of shape num_features x num_examples
        """
        return next_der * self.gradient(self.prev_x)

    def forward(self, x):
        # Activate input x (z)
        self.prev_x = x
        return self.apply_function(x)

class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'ReLU'

    def apply_function(self, z):
        """
        :type z: ndarray
        :param z: vector input

        :returns: ndarray of same length as z        
        """
        return np.maximum(0, z)

    def gradient(self, z):
        """
        :type z: ndarray
        :param z: vector input

        :returns: ndarray of same length as z        
        """
        return 1*(z > 0)

class Step(Activation):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'Step'

    def apply_function(self, z):
        treshold = 0.5
        return 1*(z>treshold)

    def gradient(self, z):
        return np.ones_like(z)

class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'Sigmoid'

    def apply_function(self, z):
        return 1/(1+np.exp(-z))

    def gradient(self, z):
        return self.apply_function(z)*(1-self.apply_function(z))

class Linear(Activation):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'Linear'

    def apply_function(self, z):
        """
        :type z: ndarray
        :param z: vector input

        :returns: ndarray of same length as z
        """
        return z

    def gradient(self, z):
        """
        :type z: ndarray
        :param z: vector input

        :returns: ndarray of same length as z        
        """
        return np.ones_like(z)


class TanH(Activation):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'TanH'

    def apply_function(self, z):
        """
        :type z: ndarray
        :param z: vector input

        :returns: ndarray of same length as z
        """
        return np.tanh(z)

    def gradient(self, z):
        """
        :type z: ndarray
        :param z: vector input

        :returns: ndarray of same length as z        
        """
        return (np.cosh(z))**(-2)


class Softmax(Activation):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'Softmax'

    def apply_function(self, z):
        """
        :type z: ndarray of shape num_classes x num_examples
        :param z: vector input 

        :returns: ndarray of shape num_classes x num_examples
        """
        z = z - np.max(z)
        exps = np.exp(z)
        return exps/np.sum(exps, axis=0)

    def jacobian(self, s):
        """
        :type s: ndarray
        :param s vector input

        :returns: ndarray of shape (len(s), len(s))
        """
        return np.diag(s) - np.outer(s, s)

    def gradient(self, z):
        """
        :type z: ndarray of shape num_classes x num_examples
        :param z: vector input

        :returns: ndarray of shape num_classes x num_classes x num_examples
        """
        s = self.apply_function(z)
        num_classes = s.shape[0]
        jacobian_tensor = np.reshape(self.jacobian(
            s[:, 0]), (1, num_classes, num_classes))
        for i in range(1, s.shape[1]):
            jacobian = np.reshape(self.jacobian(
                s[:, i]), (1, num_classes, num_classes))
            jacobian_tensor = np.append(jacobian_tensor, jacobian, axis=0)
        return jacobian_tensor
