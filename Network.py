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

    def compile(self, learning_rate, loss_type, lbda):
        """
        Sets weights and biases for all layers in network
        Sets learning rate, loss type, number of layers
        """
        # self.initialize_weights_and_biases()
        self.learning_rate = learning_rate
        self.loss_function = loss_type
        self.lbda = lbda
        self.print_network()

    def print_network(self):
        for layer in self.layers:
            print(layer, '->', end=' ')
        print('')

    def train(self, training_data, dev_data, num_classes, epochs, mini_batch_size):
        """
        :type training_data: ndarray of shape num_examples x num_features+1
        :param training_data: inputs to network horizontally stacked with targets
        """
        n = len(training_data)
        n_dev = len(dev_data)
        training_cost = []
        dev_cost = []
        for j in range(epochs):
            # Create minibatches
            mini_batches = [training_data[i:i+mini_batch_size]
                            for i in range(0, n, mini_batch_size)]
            # Train over each minibatch
            for mini_batch in mini_batches:
                self.train_batch(mini_batch, num_classes)

            # Get loss for training data after each training epoch
            train_loss = np.sum(self.get_loss(training_data, num_classes))/n
            # Get loss for dev-set after training
            dev_loss = np.sum(self.get_loss(dev_data, num_classes))/n_dev
            
            print('Epoch {} training complete, train-loss: {}, validation-loss: {}'.format(
                j, train_loss, dev_loss))
            training_cost.append(train_loss)
            dev_cost.append(dev_loss)
        return training_cost, dev_cost

    def get_loss(self, data, num_classes):
        """
        :returns: ndarray of shape num_ex x 1, loss for each example
        """
        # Get X (num_ex x input_size)
        X = data[:, :-num_classes]
        # Get Y (num_ex x output_size)
        Y = data[:, -num_classes:]
        # Forward propagation
        for layer in self.layers:
            # Activate and add X to Activation's prev_x, go through Dense
            X = layer.forward(X)
        return self.loss_function.apply_function(Y, X)

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

        # Backpropagation
        der = self.loss_function.gradient(Y, X)
        for layer in reversed(self.layers):
            # Add nabla_W and nabla_b to Dense, go through Activate
            der = layer.backpropagate(der)
        if False:
            print('out', X)
            print('Y', Y)
            print('der', der)

        # Update weights for Dense layers
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.update_weights(self.learning_rate, self.lbda)
                layer.update_biases(self.learning_rate)

        # self.print_layers()

    def print_layers(self):
        print('--- layer_details ---')
        for layer in self.layers:
            layer.print_layer_details()
