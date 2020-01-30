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
        self.num_layers = len(self.layer_sizes)

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

    def train(self, training_data, epochs=10, mini_batch_size=4):
        # Number of training examples
        n = len(training_data)
        training_cost = []
        for j in range(epochs):
            np.random.shuffle(training_data)
            # Create minibatches
            mini_batches = [training_data[i:i+mini_batch_size]
                            for i in range(0, n, mini_batch_size)]
            # Train over each minibatch
            for mini_batch in mini_batches:
                mini_batch_cost = self.update_mini_batch(mini_batch, n)
                training_cost.append(mini_batch_cost)
            print('Epoch {} training complete, loss: {}'.format(j, mini_batch_cost))
            if mini_batch_cost < 0.00002:
                print('Quit training, small loss')
                return training_cost
        return training_cost

    def update_mini_batch(self, mini_batch, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, and
        ``n`` is the total size of the training data set.
        """
        # print('mini batch: {}'.format(mini_batch))
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights_transposed]
        mini_batch_cost = 0
        # Get X (num_features x num_examples)
        X = mini_batch[:, :-1].T
        Y = mini_batch[:, -1][:, np.newaxis]
        # Forward propagation on full minibatch
        self.forward_propagation(X)
        self.print_layers()
        return 0.0000005
        for training_example in mini_batch:
            # Make x column vector
            x = training_example[:-1][:, np.newaxis]
            y = np.array([training_example[-1]])[:, np.newaxis]
            # Compute weight changes for each training case
            delta_nabla_b, delta_nabla_w, training_example_cost = self.backpropagate(
                x, y)
            # Sum all weight changes for each batch to get the gradient
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            mini_batch_cost += training_example_cost
        # Update all weights with the average gradient
        self.weights_transposed = [w-(self.learning_rate*nw)/len(mini_batch)
                                   for w, nw in zip(self.weights_transposed, nabla_w)]
        self.biases = [b-(self.learning_rate*nb)/len(mini_batch)
                       for b, nb in zip(self.biases, nabla_b)]
        return mini_batch_cost/len(mini_batch)

    def forward_propagation(self, x):
        """
        Complete forward propagation through all layers of the network.
        Set zs for all layers, except input layer
        Set activated_nodes for all layers, first activated_nodes is the input layer

        :type x: ndarray of shape num_features x num_examples
        :param x: training examples for one minibatch
        """
        # print('... forward propagation')
        activated_node = x
        # list to store all the activated nodes, layer by layer
        self.activated_nodes = [x]
        self.zs = []  # list to store all the z vectors, layer by layer
        for i, (b, w) in enumerate(zip(self.biases, self.weights_transposed)):
            print('a:', activated_node)
            print('w:', w)
            z = np.dot(w, activated_node)+b
            self.zs.append(z)
            activated_node = self.activations[i].gradient(z)
            print(activated_node.shape)
            self.activated_nodes.append(activated_node)

    def backpropagate(self, x, y):
        """
        :param x: n x m matrice, where n is the number of training examples and m is the number of features

        :returns: list of changes in weights for each layer, list of changes in biases for each layer
        """
        # Empty arrays to hold changes in each layer
        # print('x shape: {}'.format(x.shape))
        # print('y: {}'.format(y))
        nabla_b = [np.zeros_like(b) for b in self.biases]
        nabla_w = [np.zeros_like(w) for w in self.weights_transposed]
        # Forward propagation
        # self.forward_propagation(x)
        # self.print_layers()
        # Gradient descent
        # Error in loss by error in zs
        delta = (self.loss_type).gradient(
            y, self.activated_nodes[-1]) * self.activations[-1].gradient(self.zs[-1])
        # Update last bias-layer
        nabla_b[-1] = delta
        # Update last weight-layer
        nabla_w[-1] = np.dot(delta, self.activated_nodes[-2].transpose())
        # Iterate backwards
        for l in range(2, self.num_layers):
            z = self.zs[-l]
            activation_derivative = self.activations[-l].gradient(z)
            delta = np.dot(self.weights_transposed[-l+1].transpose(),
                           delta) * activation_derivative
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, self.activated_nodes[-l-1].transpose())
        prediction = self.activated_nodes[-1]
        cost = self.loss_type.apply_function(y, prediction)
        return nabla_b, nabla_w, cost

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
