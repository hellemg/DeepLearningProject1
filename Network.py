from Activation import *
import numpy as np

"""
zs: previous layer dotted with incoming weights
activated_nodes: activation function on zs
"""


class Network:
    def __init__(self, input_layer_size):
        self.layer_sizes = [input_layer_size]
        self.activations = []
        self.learning_rate = None
        self.loss_type = None
        # Number of hidden layers + 1 (for the output layer)
        self.num_layers = None
        self.biases = None
        self.weights_transposed = None
        # Values calculated in forward propagation
        self.zs = None
        self.activated_nodes = None

    def add_layer(self, num_nodes, activation):
        """
        Adds a hidden layer or an output layer
        A layer consists of the nodes, and the weights coming into
        the nodes (from the previous layer)

        Add num_nodes to the list of layers
        Adds activation object to the list of activations
        """
        self.layer_sizes.append(num_nodes)
        self.activations.append(activation)

    def compile(self, learning_rate, loss_type):
        """
        Sets weights and biases for all layers in network
        Sets learning rate, loss type, number of layers
        """
        self.initialize_weights_and_biases()
        self.learning_rate = learning_rate
        self.loss_type = loss_type
        self.num_layers = len(self.biases)
        print('Using loss:', self.loss_type)
        print('Using activations:')
        for i in range(self.num_layers):
            print(self.activations[i])
        print('# hidden layers + output layer:', self.num_layers)

    def initialize_weights_and_biases(self):
        """
        Sets biases and transposed weights for all layers, except for the first layer
        which is assumed to be an input layer. 

        :type biases: list of ndarrays, each ndarray is num_nodes x 1

        :type weights: list of ndarrays, each ndarray is num_nodes x num_nodes_prevlayer
        """
        np.random.seed(42)
        #self.biases = [np.random.randn(y) for y in self.layer_sizes[1:]]
        self.biases = [np.zeros((y,1)) for y in self.layer_sizes[1:]]
        self.weights_transposed = [np.random.normal(0, 1/np.sqrt(y), (y, x))
                                   for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]

    def train(self, training_data, num_classes=1, epochs=1, mini_batch_size=4):
        """
        Train the network on training_data with batch_size 4, 10 epochs

        :type training_data: ndarray of shape num_examples x num_features+1
        :param training_data: inputs to network horizontally stacked with targets

        :returns: list of training costs for each epoch
        """
        n = len(training_data)
        training_cost = []
        for j in range(epochs):
            # Shuffles the rows of training_data
            np.random.shuffle(training_data)
            # Create minibatches
            mini_batches = [training_data[i:i+mini_batch_size]
                            for i in range(0, n, mini_batch_size)]
            # Train over each minibatch
            for mini_batch in mini_batches:
                mini_batch_cost = self.backpropagate_batch(mini_batch, 1)
                # mini_batch_cost = self.update_mini_batch(mini_batch, num_classes)
                training_cost.append(mini_batch_cost)
            print('Epoch {} training complete, loss: {}'.format(j, mini_batch_cost))
            # If mini_batch_size=1, this needs to bed removed
            if mini_batch_cost < 0.00002:
                print('Quit training, small loss')
                return training_cost
        return training_cost

    def backpropagate_batch(self, mini_batch, num_classes, lbda=0):
        """
        Update weights and biases for all layers by applying gradient descent
        to a mini batch.

        :type mini_batch: ndarray of shape mini_batch_size x num_features+num_classes
        :param mini_batch: training data to network - inputs horizontally stacked with targets

        :type lbda: number
        :param lbda: regularization constant

        :returns: average cost for the minibatch
        """
        print('... welcome to BP batch')
        # Get X (num_features x num_examples)
        X = mini_batch[:, :-num_classes].T
        # Get Y (num_classes x num_examples)
        Y = mini_batch[:, -num_classes:].T
        # TODO: DONE Probably need to transpose X and Y, update comment with shapes
        output_layer = self.forward_propagation(X)
        loss_by_output_layer = self.loss_type.gradient(Y, output_layer)
        # TODO: Add softmax-layer
        mini_batch_size = X.shape[1]
        self.jacobi_iteration(loss_by_output_layer,
                              self.num_layers-1, mini_batch_size, lbda)
        # Return the loss
        return self.loss_type.apply_function(Y, self.activated_nodes[-1])

    def jacobi_iteration(self, loss_by_layer, layer_depth, mini_batch_size, lbda):
        """
        Updates all weights and biases in the network by Jacobi iteration.

        :type loss_by_layer: ndarray of shape mini_batch_size x num_classes
        :param loss_by_layer: change of loss in the output as a function of change in a layers nodes(??)

        :type layer_depth: int
        :param layer_depth: current layer in the network, 0 corresponds to updating first weights
        """
        print('loss by layer', loss_by_layer.shape)
        if layer_depth == 0:
            print('going into last weights')
            # c x n array
            layer_by_sum = self.activations[0].gradient(self.zs[0])
            # c x n array
            loss_by_sum = loss_by_layer * layer_by_sum
            # c x n array
            sum_by_weights = self.activated_nodes[0]
            # n x n array
            loss_by_weights = (
                (loss_by_sum) @ sum_by_weights.T)/mini_batch_size

            self.weights_transposed[0] -= self.learning_rate * \
                (loss_by_weights/mini_batch_size + lbda)
            self.biases[0] -= self.learning_rate * \
                np.sum(loss_by_sum, axis=1, keepdims=True)/mini_batch_size
        else:
            print('going into a layer')
            # c x n array
            layer_by_sum = self.activations[layer_depth].gradient(
                self.zs[layer_depth])
            # c x n array
            loss_by_sum = loss_by_layer * layer_by_sum
            # c x n array
            sum_by_weights = self.activated_nodes[layer_depth]
            # n x n array
            loss_by_weights = (
                (loss_by_sum) @ sum_by_weights.T)/mini_batch_size
            self.weights_transposed[layer_depth] -= self.learning_rate * \
                (loss_by_weights/mini_batch_size + lbda)
            self.biases[layer_depth] -= self.learning_rate * \
                np.sum(loss_by_sum, axis=1, keepdims=True)/mini_batch_size

            connecting_weights = self.weights_transposed[layer_depth]
            # Calculate new loss_by_layer to send into next round
            loss_by_layer = connecting_weights.T @ loss_by_sum
            self.jacobi_iteration(
                loss_by_layer, layer_depth-1, mini_batch_size, lbda)

    def forward_propagation(self, x):
        """
        Complete forward propagation through all layers of the network.
        Set zs for all layers, except input layer
        Set activated_nodes for all layers, first activated_nodes is the input layer

        :type x: ndarray of shape num_features x num_examples(=1)
        :param x: training examples for one minibatch

        :returns: ndarray of shape num_classes x num_examples, output of neural network
        """
        activated_node = x
        # list to store all the activated nodes, layer by layer
        self.activated_nodes = [x]
        self.zs = []  # list to store all the z vectors, layer by layer
        for i in range(len(self.biases)):
            weights = self.weights_transposed[i]
            bias = self.biases[i]
            z = (weights @ activated_node)+bias
            self.zs.append(z)
            activated_node = self.activations[i].apply_function(z)
            self.activated_nodes.append(activated_node)
        return activated_node

    def test(self, x, y):
        """
        Forward propagate x after transpose and get output, check loss
        :type x: ndarray of shape num_examples x num_features
        :param y: ndarray of shape num_classes x num_examples
        """
        validation_loss = 0
        num_examples = len(x)
        for i in range(num_examples):
            z = self.forward_propagation(x[i])
            validation_loss += self.loss_type.apply_function(y[i], z)
        return validation_loss/num_examples

    def print_layers(self):
        print('--- input nodes ---')
        print(self.activated_nodes[0])
        for i in range(len(self.zs)):
            print('*** Layer {} ***'.format(i))
            print('--- weights transposed ---')
            print(self.weights_transposed[i])
            print('--- biases ---')
            print(self.biases[i])
            print('--- zs ---')
            print(self.zs[i])
            print('--- activated nodes ---')
            print(self.activated_nodes[i+1])
