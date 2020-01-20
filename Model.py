from Function import *
import numpy as np


class Model:
    def __init__(self, learning_rate, loss_type):
        self.layers = []
        self.Function = Function()
        self.learning_rate = learning_rate
        self.loss_type = loss_type
        #self.activations = activations
        self.loss_type = None
        self.weights = None
        self.architecture = None

    def add_layer(self, no_nodes, activation, input_dim):
        weights = self.get_weights(no_nodes, input_dim)
        self.layers.append({'weights': weights,
                            'nodes': np.zeros((no_nodes, 1)),
                            'activation': activation,
                            'input_dim': input_dim})

    def compile(self, learning_rate, loss_type):
        self.loss_type = loss_type
        raise NotImplementedError

    def get_weigths(self):
        return self.weights

    def get_weights(self, no_nodes, input_dim):
        """
        Initializes transposed weight matrix on shape input dim, no_nodes
        From 0 (including) to 1 (excluding)
        """
        np.random.seed(42)
        return (np.random.rand(input_dim, no_nodes)).transpose()

    def train(self, data, labels, epochs):
        # Run FP, BP for each epoch
        for e in epochs:
            self.forward_propagation(data)
            estimated_values=self.layer[-1]['nodes']
            for i, layer in reversed(enumerate(self.layers)):
                # Get the activation-funtion-loss for this layer
                self.layers[i]['activation']
                delta_weights = self.loss_type.get_delta_w(labels, estimated_values, x, z)
                self.layers[i]['weights'] -= self.learning_rate#*'dE_i/dw_ij'

    def forward_propagation(self, x_train):
        print('... forward propagation')
        prev_x = x_train
        for i, layer in enumerate(self.layers):
            print(layer)
            z = np.matmul(layer['weights'], prev_x)
            self.layers[i]['nodes'] = layer['activation'].apply_function(z)
            print(layer)
            prev_x = layer['nodes']

    """ 
    After setting up file-reader and parser for the config-file, the first task is to build a simple neural
    network as in Figure 1. In the figure there are 5 inputs, but this obviously depends on the inputdata. As the the model
    has no hidden layers, the output based on an input x using weights w is simply
    φ(w>x + b), with some nonlinearity φ. For now, use the ReLU, φ(z) = max (0, z).
    Optimize the weights to minimize the L2-loss. Here and for all the other learning tasks, start from
    a random initialization after first setting the seed by using np.random.seed(42
    """
