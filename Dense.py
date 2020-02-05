import numpy as np


class Dense:
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

        :returns: yellow, blue, purple
        """
        # take in dL/dz (yellow + blue in textbook). prev_input = a for my cases (or x in first layer)
        self.nabla_W = self.prev_x.T @ next_der
        self.nabla_b = np.sum(next_der, axis=0)
        # Add dz/da to send to next layer
        der = next_der @ self.W.T
        return der

    def forward(self, x):
        # Calculate z for a dense layer from x (a) from previous layer
        self.prev_x = x
        return self.W.T @ x + self.b


