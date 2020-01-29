from Function import *
import numpy as np


class Model:
    def __init__(self, learning_rate, loss_type):
        self.layers = []
        self.learning_rate = learning_rate
        self.loss_type = loss_type

    def add_layer(self, no_nodes, activation, input_dim):
        weights = self.get_weights(no_nodes, input_dim)
        self.layers.append({'weights_transposed': weights,
                            'nodes': np.zeros((no_nodes, 1)),
                            'activation': activation,
                            'input_dim': input_dim})

    def get_weights(self, no_nodes, input_dim):
        """
        Initializes transposed weight matrix on shape input dim, no_nodes with the Xavier initialization
        """
        np.random.seed(42)
        return np.random.normal(0, 1/(np.sqrt(input_dim)), (input_dim, no_nodes)).transpose()

    def print_layers():
        for layer in self.layers:
            print('weights:', layer['weights_transposed'])
            print('nodes', layer['nodes'])
            print('activation', layer['activation'])
            print('input dim', layer['input_dim'])

    def train(self, inputs, targets, epochs):
        # Add layer for inputs-nodes
        self.layers.insert(0, {'weights_transposed': None,
                               'nodes': inputs,
                               'activation': None,
                               'input_dim': None})
        # Run FP, BP for each epoch
        for e in range(epochs):
            self.forward_propagation()
            self.print_layers()
            print('... backpropagation')
            output_values = self.layers[-1]['nodes']
            assert output_values.shape == targets.shape
            # Scalar error from NN
            output_errors = self.loss_type.apply_function(
                y=targets, z=output_values)
            if np.abs(output_errors) < 0.0000002:
                print('stop training on round', e,
                      'loss is', output_errors)
                return
            print('loss:', output_errors)
            J_loss_by_output = self.loss_type.gradient(
                y=targets, z=output_values)
            self.layers[1]['weights_transposed'] += self.learning_rate * \
                np.transpose(J_loss_by_output)
            print('new weights:', self.layers[1]['weights_transposed'])

    def jacobian_iteration(self, targets, soft_max_model=True):
        # Values estimated by NN
        output_values = self.layers[-1]['nodes']
        # Effect of input to layer on output of layer ()
        J_output_layer_by_sum = self.layers[-1]['activation'].derivative(
            output_values)
        # R = np.multiply(np.identity(output_size), J_output_layer_by_sum)

        # NOT JUST FOR FIRST WEIGHTS; BUT ITERATIVE

        # Change in a layer's nodes by the earlier layer's nodes (Z by Y)
        J_layer_by_earlier_layer = J_output_layer_by_sum @ self.layers[-1]['weights_transposed']
        # Values of previours layer (Y)
        earlier_layer = self.layers[-2]['nodes']
        J_layer_by_incoming_weights_simplified = np.outer(
            earlier_layer, J_output_layer_by_sum)
        # Change in loss by change of output values (L by Z)
        J_loss_by_output = self.loss_type.gradient(y=targets, z=output_values)
        # Initialize before iteration
        J_loss_by_layer = J_loss_by_output
        if soft_max_model:
            # Get softmax jacobian of outputvalues z (S by Z)
            J_softmax_by_output = self.layers[-1]['activation'].jacobian(
                output_values)
            # Add softmax-layer to jacobi-iteration (L by Z, S between)
            J_loss_by_output = J_loss_by_softmax @ J_softmax_by_output
        # Jacobi-iteration
        for i, layer in enumerate(reversed(self.layers)):
            # Calculate values for updating weights
            J_loss_by_input_weights = J_loss_by_layer * \
                J_layer_by_incoming_weights_simplified
            # TODO: Update for earlier layer
            J_layer_by_incoming_weights_simplified = 3
            # Calculate values for further iterations
            J_loss_by_layer = J_loss_by_layer @ J_layer_by_earlier_layer
            input_sum_to_layer = layer['weights'] @ self.layers[len(
                self.layers) - 1 - i]['nodes']
            J_layer_by_sum = layer['activation'].derivative(input_sum_to_layer)
            J_layer_by_earlier_layer = J_layer_by_sum @ layer['weights_transposed']
            # TODO: Stop iteration before first layer

        # Diagonal matrix with output derivatives (slide 48, lecture 2)
        # TODO: Update W with new weights

    def forward_propagation(self):
        print('... forward propagation')
        for i, layer in enumerate(self.layers[:-1]):
            z = np.matmul(self.layers[i+1]
                          ['weights_transposed'], layer['nodes'])
            self.layers[i+1]['nodes'] = self.layers[i +
                                                    1]['activation'].apply_function(z)
