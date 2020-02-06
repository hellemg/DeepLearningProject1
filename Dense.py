from Function import *
import numpy as np


class Dense(Function):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.initialize_weights(input_dims, output_dims)

    def __str__(self):
        return 'Dense ({}, {})'.format(self.W.shape[0], self.W.shape[1])

    def initialize_weights(self, input_dims, output_dims):
        # Set bias and weights
        self.b = np.zeros((1, output_dims))
        self.W = np.random.normal(
            0, np.sqrt(2/(input_dims + output_dims)), (input_dims, output_dims))

    def update_weights(self, learning_rate, lbda):
        self.W = self.W - learning_rate*(self.nabla_W + lbda*self.W)

    def update_biases(self, learning_rate):
        self.b = self.b - learning_rate*self.nabla_b

    def backpropagate(self, next_der):
        # backpropagate partial derivative from next layer
        # take in dL/dz (yellow + blue in textbook). prev_input = a for my cases (or x in first layer)
        self.nabla_W = self.prev_x.T @ next_der
        self.nabla_b = np.sum(next_der, axis=0, keepdims=True)
        # Add dz/da (purple) to send to next layer
        der = next_der @ self.W.T
        return der

    def forward(self, x):
        # Calculate z for a dense layer from x (a) from previous layer
        self.prev_x = x
        z = (x @ self.W) + self.b
        return z

    def print_layer_details(self):
        print('**',self)
        print('-input')
        print(self.prev_x)
        print('-nabla_W')
        print(self.nabla_W)
        print('-nabla_b')
        print(self.nabla_b)
        print('-W')
        print(self.W)
        print('-b')
        print(self.b)