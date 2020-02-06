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
            print(layer, '->' )

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
        Train the network on training_data with batch_size 4, 10 epochs

        :type training_data: ndarray of shape num_examples x num_features+1
        :param training_data: inputs to network horizontally stacked with targets

        :returns: list of training costs for each epoch
        """
        n = len(training_data)
        training_cost = []
        for j in range(epochs):
            # Shuffles the rows of training_data
            #np.random.shuffle(training_data)
            # Create minibatches
            mini_batches = [training_data[i:i+mini_batch_size]
                            for i in range(0, n, mini_batch_size)]
            # Train over each minibatch
            for mini_batch in mini_batches:
                self.train_batch(mini_batch, num_classes)    
            
                # mini_batch_cost = self.backpropagate_batch(
                #     mini_batch, num_classes, lbda)
                # # mini_batch_cost = self.update_mini_batch(mini_batch, num_classes)
                # training_cost.append(mini_batch_cost)
            #print('Epoch {} training complete, loss: {}'.format(j, mini_batch_cost))

    def train_batch(self, mini_batch, num_classes):
        # Get X (num_ex x input_size)
        X = mini_batch[:, :-num_classes]
        # Get Y (num_ex x output_size)
        Y = mini_batch[:, -num_classes:]

        # Forward propagation
        for layer in self.layers:
            # Activate and add X to Activation's prev_x, go through Dense
            X = layer.forward(X)
        
        # Backpropagation
        der = self.loss_function.apply_function(Y, X)
        for layer in reversed(self.layers):
            # Add nabla_W and nabla_b to Dense, go through Activate
            der = layer.backpropagate(der)

        # Update weights for Dense layers
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.update_weights(self.learning_rate, self.lbda)
                layer.update_biases(self.learning_rate)



    def backpropagate_batch(self, mini_batch, num_classes, lbda):
        """
        Update weights and biases for all layers by applying gradient descent
        to a mini batch.

        :type mini_batch: ndarray of shape mini_batch_size x num_features+num_classes
        :param mini_batch: training data to network - inputs horizontally stacked with targets

        :type lbda: number
        :param lbda: regularization constant

        :returns: average cost for the minibatch
        """
        # Get X (num_features x num_examples)
        X = mini_batch[:, :-num_classes].T
        # Get Y (num_classes x num_examples)
        Y = mini_batch[:, -num_classes:].T
        mini_batch_size = X.shape[1]
        output_layer = self.forward_propagation(X)
        loss_by_output_layer = self.loss_function.gradient(Y, output_layer)
        self.backpropagate_output(
            loss_by_output_layer, self.num_layers-1, mini_batch_size, lbda)
        # Return the loss
        #self.print_layers()
        # print(self.activated_nodes[-1])
        # print(Y)
        # print(self.activated_nodes[-1].shape)
        # print(Y.shape)
        # input()
        return self.loss_function.apply_function(Y, self.activated_nodes[-1])

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

    def jacobi_iteration(self, loss_by_layer, layer_depth, mini_batch_size, lbda):
        """
        Updates all weights and biases in the network by Jacobi iteration.

        :type loss_by_layer: ndarray of shape num_classes x mini_batch_size
        :param loss_by_layer: change of loss in the output as a function of change in a layers nodes(??)

        :type layer_depth: int
        :param layer_depth: current layer in the network, 0 corresponds to updating first weights

        :type mini_batch_size: int
        :param mini_batch_size: number of exampes in batch

        :type lbda: float
        :param lbda: regularization constant
        """
        # print('loss by layer', loss_by_layer.shape)
        # print('going into last weights')
        # print('layer_depth', layer_depth)
        # num_classes x num_examples
        layer_by_sum = self.activations[layer_depth].gradient(
            self.zs[layer_depth])
        # print(layer_by_sum.shape)
        # num_classes x num_examples
        loss_by_sum = loss_by_layer * layer_by_sum
        # print(loss_by_sum.shape)
        # num_classes_prev x num_examples
        sum_by_weights = self.activated_nodes[layer_depth]
        # print('sum by weights', sum_by_weights.shape)
        # num_classes x num_classes_prev
        loss_by_weights = (
            (loss_by_sum) @ sum_by_weights.T)/mini_batch_size
        # print(loss_by_weights.shape)
        # print(self.weights_transposed[layer_depth].shape)
        # print(self.biases[layer_depth].shape)
        # print(np.sum(loss_by_sum, axis=1, keepdims=True).shape)
        self.weights_transposed[layer_depth] -= self.learning_rate * \
            (loss_by_weights+lbda)/mini_batch_size
        self.biases[layer_depth] -= self.learning_rate * \
            np.sum(loss_by_sum, axis=1, keepdims=True)/mini_batch_size
        if layer_depth != 0:
            # print('going into a layer')
            connecting_weights = self.weights_transposed[layer_depth]
            # Calculate new loss_by_layer to send into next round
            loss_by_layer = connecting_weights.T @ loss_by_sum
            # print('connecting_weights', connecting_weights.shape)
            # print('loss_by_sum', loss_by_sum.shape)
            # print('next loss by layer:', loss_by_layer.shape)
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
        activated_node=x
        # list to store all the activated nodes, layer by layer
        self.activated_nodes=[x]
        self.zs=[]  # list to store all the z vectors, layer by layer
        for i in range(len(self.biases)):
            weights=self.weights_transposed[i]
            bias=self.biases[i]
            z=(weights @ activated_node)+bias
            self.zs.append(z)
            activated_node=self.activations[i].apply_function(z)
            self.activated_nodes.append(activated_node)
        return activated_node

    def test(self, test_data, num_classes):
        """
        Forward propagate x after transpose and get output, check loss
        :type test_data: ndarray of shape num_examples x num_features+num_labels
        :param test_data: examples horizontally stacked with labels

        :type num_classes: int
        :param num_classes: number of output nodes
        """
        validation_loss=0
        # Get X (num_features x num_examples)
        X = test_data[:, :-num_classes].T
        # Get Y (num_classes x num_examples)
        Y = test_data[:, -num_classes:].T
        Z = self.forward_propagation(X)
        return self.loss_function.apply_function(Y, Z)

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
