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

    def jacobian_iteration(self):
        # Size of output layer TODO: check 0 or 1 index
        output_size = self.layers[-1]['weights'].shape[0]
        estimated_values = self.layers[-1]['nodes']
        J_output_layer_by_sum = self.layers[-1]['activation'].derivative(estimated_values)
        R = np.multiply(np.identity(output_size), J_output_layer_by_sum)
        # Add stuff for softmax

        # NOT JUST FOR FIRST WEIGHTS; BUT ITERATIVE
        # TODO: Rename 'weights' to 'weights_transposed'
        J_layer_by_earlier_layer = J_output_layer_by_sum @ self.layers[-1]['weights']
        earlier_layer = self.layers[-2]['nodes']
        J_layer_by_incoming_weights_simplified = np.outer(earlier_layer, J_output_layer_by_sum)
        print('Shape of J_Z_by_W_simplified should be len(y) times len(z):', J_layer_by_incoming_weights_simplified.shape)
        for layer in reversed(self.layers):
            R = R @ J_layer_by_earlier_layer
            # Check shape
            # Break before input, maybe not needed?
        # Get J_first_layer_by_first_weights
        J_loss_by_first_weights = 3 #R @ J_first_layer_by_first_weights
        # Update first weights


        # Diagonal matrix with output derivatives (slide 48, lecture 2)
        # TODO: Get jacobian of Loss by Z (output)
        # TODO: Get jacobian of Z by W: J_layer_by_incoming_weights_simplified
        # TODO: Get Jacobian of Loss by W (weights from Y to Z) from the two above. See slides 52 for numpy
        # TODO: Update W with the above

        # TODO: Get jacobian of Z by Y
        # TODO: Get jacobian of L by Y, continue iteration

        # TODO: Get Y, output from layer Y (vector [y1, y2, ..., ,n])
        # TODO: Y op diag(R) (op = outer product. slide 50)


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
