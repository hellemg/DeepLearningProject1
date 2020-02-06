import numpy as np
from Function import *

class Activation(Function):
    def __init__(self):
        super().__init__()

    def backpropagate(self, next_der, debug=False):
        # backpropagate partial derivative from next layer 
        der = next_der * self.gradient(self.prev_x)
        if debug:
            print('input der A', next_der)
            print('prev input A (z)', self.prev_x)
            print('gradient of prev input A (z)', self.gradient(self.prev_x))
            print('der', der)
        return der

    def forward(self, x, debug=False):
        # Activate input x (z)
        self.prev_x = x
        if debug:
            print('input:', x)
        return self.apply_function(x)

    def print_layer_details(self):
        print('**',self)
        print('-input')
        print(self.prev_x)

class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'ReLU'

    def apply_function(self, z):
        return np.maximum(0, z)

    def gradient(self, z):
        return 1.0*(z > 0)

class Linear(Activation):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'Linear'

    def apply_function(self, z):
        return z

    def gradient(self, z):
        return np.ones_like(z)


class TanH(Activation):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'TanH'

    def apply_function(self, z):
        return np.tanh(z)

    def gradient(self, z):
        return (np.cosh(z))**(-2)

class Softmax(Activation):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'Softmax'

    def apply_function(self, z):
        """
        :type z: ndarray of shape num_ex x input_size

        :returns: ndarray of shape num_ex x output_size (output_size = input_size for Activation)
        """
        z = z - np.max(z)
        exps = np.exp(z)
        return exps/np.sum(exps, axis=1, keepdims=True)

    def jacobian(self, s):
        """
        make jacobian for one example
        """
        return np.diag(s) - np.outer(s, s)

    def gradient(self, z):
        """
        Takes in weighted sum from prev layer
        """
        s = self.apply_function(z)
        num_classes = s.shape[1]
        jacobian = self.jacobian(s[0])
        jacobian_tensor = np.reshape(jacobian, (1, num_classes, num_classes))
        for i in range(1, s.shape[0]):
            jacobian = self.jacobian(s[i])
            jacobian = np.reshape(jacobian, (1, num_classes, num_classes))
            jacobian_tensor = np.append(jacobian_tensor, jacobian, axis=0)
        return jacobian_tensor

    def backpropagate(self, next_der, debug=False):
        next_der = np.reshape(next_der, (next_der.shape[0], next_der.shape[1], 1))
        der = self.gradient(self.prev_x) @ next_der
        return np.reshape(der, (next_der.shape[0], self.prev_x.shape[1]))