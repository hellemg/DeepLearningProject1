from Activation import *
import numpy as np


class Model:
    def __init__(self, learning_rate, loss_type):
        self.layers = [[]]
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

    def print_layers(self):
        for layer in self.layers:
            print('weights_transposed:', layer['weights_transposed'])
            print('nodes', layer['nodes'])
            print('activation', layer['activation'])
            print('input dim', layer['input_dim'])

    def train(self, inputs, targets, epochs=100):
        # Add layer for inputs-nodes
        self.layers[0] = {'weights_transposed': None,
                          'nodes': inputs,
                          'activation': None,
                          'input_dim': None}
        # Run FP, BP for each epoch
        for e in range(epochs):
            self.forward_propagation()
            self.print_layers()
            prev_output_error = self.backpropagation(targets)
            if np.abs(prev_output_error) < 0.0000002:
                print('stop training on round', e,
                      'loss is', prev_output_error)
                return

    def backpropagation(self, targets, softmax_model=False):
        print('... backpropagation')
        output_values = self.layers[-1]['nodes']
        #assert output_values.shape == targets.shape
        # Scalar error from NN
        output_errors = self.loss_type.apply_function(
            targets, output_values)
        print('loss:', output_errors)
        # Change in loss by change in output layer (L by Z)
        J_loss_by_layer = self.loss_type.gradient(targets, output_values)
        if softmax_model:
            # TODO: Add J_loss_by_softmax
            J_loss_by_softmax = 3
            # Get softmax jacobian of output values z (S by Z)
            J_softmax_by_output = self.layers[-1]['activation'].jacobian(
                output_values)
            # Add softmax-layer to jacobi-iteration (L by Z, S between)
            J_loss_by_layer = J_loss_by_softmax @ J_softmax_by_output
        # For all but first layer (first layer only have inputs)
        for i, layer in enumerate(reversed(self.layers[1:])):
            # Change in a layer's nodes by the earlier layer's nodes (Z by Y)
            J_layer_by_sum = layer['activation'].gradient(layer['nodes'])
            J_layer_by_earlier_layer = J_layer_by_sum @ layer['weights_transposed']
            J_loss_by_earlier_layer = J_loss_by_layer @ J_layer_by_earlier_layer

            J_layer_by_weights = np.outer(
                self.layers[len(self.layers) - i - 1]['nodes'], np.diag(J_layer_by_sum))
            J_loss_by_weigths = J_loss_by_layer * J_layer_by_weights
            layer['weights_transposed'] += self.learning_rate*J_loss_by_weigths

            #print('L by Sum: {}'.format(J_layer_by_sum))
            #print('Z by Y: {}'.format(J_layer_by_earlier_layer))
            #print('L by Y: {}'.format(J_loss_by_layer))
            # print('-----------')
            #print('Z by W: {}'.format(J_layer_by_weights))
            #print('L by W: {}'.format(J_loss_by_weigths))
            # Update for the next round
            J_loss_by_layer = J_loss_by_earlier_layer
        return output_errors

    def forward_propagation(self):
        print('... forward propagation')
        for i, layer in enumerate(self.layers[1:]):
            # Index with i, which is 0 when we are at layer 1
            z = np.matmul(layer['weights_transposed'], self.layers[i]['nodes'])
            layer['nodes'] = layer['activation'].apply_function(z)
