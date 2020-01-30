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

    def initialize_weights_and_biases(self):
        """
        Sets biases and transposed weights for all layers, except for the first layer
        which is assumed to be an input layer. 

        :type biases: list of ndarrays, each ndarray is num_nodes x 1

        :type weights: list of ndarrays, each ndarray is num_nodes x num_nodes_prevlayer
        """
        np.random.seed(42)
        self.biases = [np.random.randn(y, 1) for y in self.layer_sizes[1:]]
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
                mini_batch_cost = self.update_mini_batch(
                    mini_batch, num_classes)
                training_cost.append(mini_batch_cost)
            print('Epoch {} training complete, loss: {}'.format(j, mini_batch_cost))
            # If mini_batch_size=1, this needs to bed removed
            if mini_batch_cost < 0.00002:
                print('Quit training, small loss')
                return training_cost
        return training_cost

    def update_mini_batch(self, mini_batch, num_classes):
        """
        Update weights and biases for all layers by applying gradient descent
        to a mini batch. Both are updated with the average gradient for each
        training example.

        :type mini_batch: ndarray of shape mini_batch_size x num_features+1
        :param mini_batch: inputs to network horizontally stacked with targets

        :returns: average cost for the minibatch
        """
        mini_batch_size = mini_batch.shape[0]
        #print('mini batch: {}'.format(mini_batch))
        nabla_b = [np.zeros_like(b) for b in self.biases]
        nabla_w = [np.zeros_like(w) for w in self.weights_transposed]
        mini_batch_cost = 0
        # Get X (num_features x num_examples)
        X = mini_batch[:, :-num_classes].T
        Y = mini_batch[:, -num_classes:]
        # Forward propagation on full minibatch
        #self.forward_propagation(X) 
        # self.print_layers()
        for i in range(mini_batch_size):
            # Make x column vector
            x = X[:, i][:, np.newaxis]
            y = Y[i][:, np.newaxis]
            # Compute weight changes, bias changes, and loss for each training case
            delta_nabla_b, delta_nabla_w, training_example_cost = self.backpropagate(
                x, y)
            # Sum weight changes in each layer
            for r in range(self.num_layers):
                print(delta_nabla_b[r].shape)
                print(nabla_b[r].shape)
                nabla_w[r] += delta_nabla_w[r]
                nabla_b[r] += delta_nabla_b[r]
            mini_batch_cost += training_example_cost
        # Update all weights and biases with the average gradient
        for hl in range(self.num_layers):
            self.weights_transposed[hl] -= (self.learning_rate *
                                            nabla_w[hl])/mini_batch_size
            self.biases[hl] -= (self.learning_rate*nabla_b[hl])/mini_batch_size
        return mini_batch_cost/len(mini_batch)

    def forward_propagation(self, x):
        """
        Complete forward propagation through all layers of the network.
        Set zs for all layers, except input layer
        Set activated_nodes for all layers, first activated_nodes is the input layer

        :type x: ndarray of shape num_features x num_examples(=1)
        :param x: training examples for one minibatch

        :returns: ndarray of shape num_classes x num_examples, output of neural network
        """
        # print('... forward propagation')
        activated_node = x
        # list to store all the activated nodes, layer by layer
        self.activated_nodes = [x]
        self.zs = []  # list to store all the z vectors, layer by layer
        for i, (b, w) in enumerate(zip(self.biases, self.weights_transposed)):
            z = np.dot(w, activated_node)+b
            self.zs.append(z)
            activated_node = self.activations[i].apply_function(z)
            self.activated_nodes.append(activated_node)
        return activated_node

    def backpropagate(self, x, y):
        """
        :type x: ndarray of shape num_features x num_examples(==1)
        :param x: training example

        :type y: ndarray of shape num_classes x num_examples(==1)
        :param y: target

        :returns: list of changes in weights for each layer, list of changes
        in biases for each layer, cost for training example
        """
        # Empty arrays to hold changes in each layer
        # print('x shape: {}'.format(x.shape))
        # print('y: {}'.format(y))
        nabla_b = [np.zeros_like(b) for b in self.biases]
        nabla_w = [np.zeros_like(w) for w in self.weights_transposed]
        # Forward propagation
        output_layer = self.forward_propagation(x).T[0]
        y = y.T[0]
        loss_gradient = self.loss_type.gradient(y, output_layer)
        activation_gradient = self.activations[-1].gradient(self.zs[-1]).T[0]
        delta = np.multiply(loss_gradient, activation_gradient)
        # Update last bias-layer
        nabla_b[-1] = delta[:, np.newaxis]
        # Update last weight-layer
        previous_a = self.activated_nodes[-2]
        print('delta:',delta.shape)
        print('prev a:',previous_a.shape)
        print(self.weights_transposed[-1].shape)

        # TODO: find combinations of these nabla_w[-1] = previous_a, delta
        # previous layer does not have same shape as last layer

        print('---------')
        # Iterate backwards
        for l in range(2, self.num_layers+1):
            z = self.zs[-1].T[0]
            activation_gradient = self.activations[-l].gradient(z)

            # TODO: delta = (self.weights_transposed[-l+1], delta) * activation_gradient

            # TODO: nabla_b[-l] = delta
            previous_a = self.activated_nodes[-l-1]

            # TODO: nabla_w[-l] = delta, self.activated_nodes[-l-1]

        prediction = self.activated_nodes[-1].T
        for pred in prediction:
            print(y)
            print(pred)
            cost = self.loss_type.apply_function(y, pred)
        return nabla_b, nabla_w, cost

    def test(self, x, y):
        print('___ testing')
        """
        Forward propagate x after transpose and get output, check loss
        :type x: ndarray of shape num_examples x num_features
        :param y: ndarray of shape num_classes x num_examples
        """
        # TODO: Fix test
        validation_loss = 0
        for x_ex in x:
            z = self.forward_propagation(x_ex)
            validation_loss += self.loss_type.apply_function(y.T, z)
            print(z)
            print(y.T)
            print()
        return validation_loss/x.shape[0]

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
