from Function import *
import numpy as np


class Model:
    def __init__(self, learning_rate, loss_type):
        self.layers = []
        self.Function = Function()
        self.learning_rate = learning_rate
        self.loss_type = loss_type
        #self.activations = activations
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

    def train(self, inputs, targets, epochs):
        # Run FP, BP for each epoch
        for e in range(epochs):
            self.forward_propagation(inputs)
            print(self.layers)
            print(inputs.shape)
            return
            print('... backpropagation')
            estimated_values = self.layers[-1]['nodes']
            assert estimated_values.shape == targets.shape
            output_errors = estimated_values - targets
            if np.sum(output_errors) < 0.0000002:
                print('stop training on round',e,'loss is', output_errors)
                return
            print('loss:', output_errors)
            error_change = inputs*output_errors
            print(error_change)
            self.layers[0]['weights'] -= self.learning_rate * \
                np.transpose(error_change)
            print('new weights:', self.layers[0]['weights'])

    def forward_propagation(self, x_train):
        print('... forward propagation')
        prev_x = x_train
        for i, layer in enumerate(self.layers):
            z = np.matmul(layer['weights'], prev_x)
            self.layers[i]['nodes'] = layer['activation'].apply_function(z)
            prev_x = layer['nodes']

    """ 
    After setting up file-reader and parser for the config-file, the first task is to build a simple neural
    network as in Figure 1. In the figure there are 5 inputs, but this obviously depends on the inputdata. As the the model
    has no hidden layers, the output based on an input x using weights w is simply
    φ(w>x + b), with some nonlinearity φ. For now, use the ReLU, φ(z) = max (0, z).
    Optimize the weights to minimize the L2-loss. Here and for all the other learning tasks, start from
    a random initialization after first setting the seed by using np.random.seed(42
    """
