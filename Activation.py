import numpy as np
from Function import *

class Activation(Function):
    def __init__(self):
        super().__init__()

    def backpropagate(self, next_der):
        # backpropagate partial derivative from next layer 
        return next_der * self.gradient(self.prev_x)

    def forward(self, x):
        # Activate input x (z)
        self.prev_x = x
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
        raise NotImplementedError
        z = z - np.max(z)
        exps = np.exp(z)
        return exps/np.sum(exps, axis=0)

    def jacobian(self, s):
        return np.diag(s) - np.outer(s, s)

    def gradient(self, z):
        raise NotImplementedError
        s = self.apply_function(z)
        num_classes = s.shape[0]
        jacobian_tensor = np.reshape(self.jacobian(
            s[:, 0]), (1, num_classes, num_classes))
        for i in range(1, s.shape[1]):
            jacobian = np.reshape(self.jacobian(
                s[:, i]), (1, num_classes, num_classes))
            jacobian_tensor = np.append(jacobian_tensor, jacobian, axis=0)
        return jacobian_tensor
