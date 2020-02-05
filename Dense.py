from Function import *
import numpy as np


class Dense(Function):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.initialize_weights(input_dims, output_dims)
    
    def __str__(self):
        return 'Dense'

    def initialize_weights(self, input_dims, output_dims):
        # Set bias and weights
        self.b = np.zeros((output_dims, 1))
        self.W = np.random.normal(0, 1/np.sqrt(output_dims), (input_dims, output_dims))

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


