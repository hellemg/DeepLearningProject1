from Activation import *
import numpy as np
from Dense import *
"""
zs: previous layer dotted with incoming weights
activated_nodes: activation function on zs
"""


class Network:
    def __init__(self):
        # Add Activation and Dense
        self.layers = []

        self.learning_rate = None
        self.loss_function = None

    def add_layer(self, function):
        """
        Adds a hidden layer or an output layer
        A layer consists of the nodes, and the weights coming into
        the nodes (from the previous layer)

        Add num_nodes to the list of layers
        Adds activation object to the list of activations
        """
        self.layers.append(function)
        #self.layer_sizes.append(num_nodes)
        #self.activations.append(activation)

    def compile(self, learning_rate, loss_type, lbda):
        """
        Sets weights and biases for all layers in network
        Sets learning rate, loss type, number of layers
        """
        #self.initialize_weights_and_biases()
        self.learning_rate = learning_rate
        self.loss_function = loss_type
        self.lbda = lbda
        self.print_network()

    def print_network(self):
        for layer in self.layers:
            print(layer, '->', end=' ')
        print('')

    def initialize_weights_and_biases(self):
        """
        Sets biases and transposed weights for all layers, except for the first layer
        which is assumed to be an input layer.

        :type biases: list of ndarrays, each ndarray is num_nodes x 1

        :type weights: list of ndarrays, each ndarray is num_nodes x num_nodes_prevlayer
        """
        # np.random.seed(42)
        # self.biases = [np.zeros((y, 1)) for y in self.layer_sizes[1:]]
        # np.random.seed(42)
        # self.weights_transposed = [np.random.normal(0, 1/np.sqrt(y), (y, x))
        #                            for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]

    def train(self, training_data, num_classes, epochs, mini_batch_size):
        """
        :type training_data: ndarray of shape num_examples x num_features+1
        :param training_data: inputs to network horizontally stacked with targets
        """
        n = len(training_data)
        training_cost = []
        epoch_losses = 0
        for j in range(epochs):
            # Create minibatches
            mini_batches = [training_data[i:i+mini_batch_size]
                            for i in range(0, n, mini_batch_size)]
            # Train over each minibatch
            for mini_batch in mini_batches:
                mini_batch_loss_before_BP = self.train_batch(mini_batch, num_classes) 
                epoch_losses += mini_batch_loss_before_BP
            
            epoch_losses = np.sum(epoch_losses)/n
            print('Epoch {} training complete, loss before training: {}'.format(j, epoch_losses))
        
    def train_batch(self, mini_batch, num_classes):
        """
        :returns: ndarray of shape num_ex x 1 - loss for each example
        """
        # Get X (num_ex x input_size)
        X = mini_batch[:, :-num_classes]
        # Get Y (num_ex x output_size)
        Y = mini_batch[:, -num_classes:]

        # Forward propagation
        for layer in self.layers:
            # Activate and add X to Activation's prev_x, go through Dense
            X = layer.forward(X)
                
        mini_batch_loss_before_BP = self.loss_function.apply_function(Y, X)

        # Backpropagation
        der = self.loss_function.gradient(Y, X)
        for layer in reversed(self.layers):
            # Add nabla_W and nabla_b to Dense, go through Activate
            der = layer.backpropagate(der)
        if False:
            print('out', X)
            print('Y', Y)
            print('der', der)

        # ABOVE THIS IS OK
        # Update weights for Dense layers
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.update_weights(self.learning_rate, self.lbda)
                layer.update_biases(self.learning_rate)
            
        return mini_batch_loss_before_BP
        # self.print_layers()

    def backpropagate_output(self, loss_by_output_layer, layer_depth, mini_batch_size, lbda):
        """
        Compute output layer
        """
        if isinstance(self.activations[-1], Softmax):
            loss_by_output_layer = loss_by_output_layer.T
            # print('loss by output layer:', loss_by_output_layer.shape)
            # num_examples x num_classes x 1
            loss_by_output_layer = np.reshape(
                loss_by_output_layer, (mini_batch_size, loss_by_output_layer.shape[1], 1))
            # print('loss_by_output_layer', loss_by_output_layer.shape)
            # num_examples x num_classes x num_classes
            softmax_by_layer = self.activations[-1].gradient(
                self.activated_nodes[-1])
            # print('softmax_by_layer', softmax_by_layer.shape)
            # num_examples x num_classes x 1
            loss_by_output_layer = softmax_by_layer @ loss_by_output_layer
            # print('loss by output layer:', loss_by_output_layer.shape)
            # num_examples x num_classes x num_classes
            layer_by_sum = self.activations[-1].gradient(self.zs[-1])
            # print('layer by sum:', layer_by_sum.shape)
            # num_examples x num_classes
            loss_by_sum = np.reshape(
                layer_by_sum @ loss_by_output_layer, (mini_batch_size, layer_by_sum.shape[1]))
            # print('loss by sum:', loss_by_sum.shape)
            # num_classes_prev x num_examples
            sum_by_weights = self.activated_nodes[-2]
            # print('sum_by_weights', sum_by_weights.shape)
            # num_classes x num_classes_prev
            loss_by_weights = (
                (sum_by_weights @ loss_by_sum)/mini_batch_size).T
            # print('loss_by_weights', loss_by_weights.shape)
            # print('weights:', self.weights_transposed[layer_depth].shape)
            self.weights_transposed[layer_depth] -= self.learning_rate * \
                (loss_by_weights+lbda)/mini_batch_size
            # print(np.sum(loss_by_sum.T, axis=1, keepdims=True).shape)
            # print('biases:', self.biases[layer_depth].shape)
            self.biases[layer_depth] -= self.learning_rate * \
                np.sum(loss_by_sum.T, axis=1, keepdims=True) / \
                mini_batch_size
            if layer_depth != 0:
                connecting_weights = self.weights_transposed[layer_depth]
                # Calculate new loss_by_layer to send into next round
                # print('connecting_weights', connecting_weights.shape)
                # print('loss_by_sum', loss_by_sum.shape)
                loss_by_layer = connecting_weights.T @ loss_by_sum.T
                self.jacobi_iteration(
                    loss_by_layer, layer_depth-1, mini_batch_size, lbda)
        else:
            self.jacobi_iteration(loss_by_output_layer,
                                  layer_depth, mini_batch_size, lbda)

    def print_layers(self):
        print('--- layer_details ---')
        for layer in self.layers:
            layer.print_layer_details()
    