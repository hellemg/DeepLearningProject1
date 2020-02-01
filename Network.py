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

    def initialize_weights_and_biases(self):
        """
        Sets biases and transposed weights for all layers, except for the first layer
        which is assumed to be an input layer. 

        :type biases: list of ndarrays, each ndarray is num_nodes x 1

        :type weights: list of ndarrays, each ndarray is num_nodes x num_nodes_prevlayer
        """
        np.random.seed(42)
        self.biases = [np.random.randn(y) for y in self.layer_sizes[1:]]
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
        # Get X (num_examples x num_features)
        X = mini_batch[:, :-num_classes]
        # Get Y (num_examples x num_classes)
        Y = mini_batch[:, -num_classes:]
        # TODO: Probably need to transpose X and Y, update comment with shapes
        output_layer = self.forward_propagation(X)
        loss_by_output_layer = self.loss_type.gradient(Y, output_layer)
        # TODO: Check shape of loss_by_layer, ensure that gradient methods outputs correct shape
        # TODO: Add softmax-layer
        mini_batch_size = 20
        # TODO: Add mini_batch_size after transposing X
        self.jacobi_iteration(loss_by_output_layer, self.num_layers-1, mini_batch_size)
        # Return the loss
        # TODO: Check shape of loss, sum and divide by mini_batch_size if necessary
        return self.loss_type.apply_function(self.activated_nodes[-1])

    def jacobi_iteration(self, loss_by_layer, layer_depth, mini_batch_size):
        """
        Updates all weights and biases in the network by Jacobi iteration.

        :type loss_by_layer: ndarray of shape mini_batch_size x num_classes
        :param loss_by_layer: change of loss in the output as a function of change in a layers nodes(??)

        :type layer_depth: int
        :param layer_depth: current layer in the network, 0 corresponds to updating first weights
        """
        if layer_depth == 0:
            # TODO: layer_by_weights = self.activated_nodes[0] outer self.activations[0].gradient()
            # TODO: loss_by_weights = loss_by_layer x layer_by_weights
            # TODO: self.weights_transposed[0] -= self.learning_rate*loss_by_weights/mini_batch_size
            # TODO: self.biases[0] -= ....
        else:
            # ALl of the above?
            # TODO: layer_by_layer

    def update_mini_batch(self, mini_batch, num_classes, lbda=0):
        """
        Update weights and biases for all layers by applying gradient descent
        to a mini batch. Both are updated with the average gradient for each
        training example.

        :type mini_batch: ndarray of shape mini_batch_size x num_features+num_classes
        :param mini_batch: inputs to network horizontally stacked with targets

        :type lbda: number
        :param lbda: regularization constant

        :returns: average cost for the minibatch
        """
        mini_batch_size = mini_batch.shape[0]
        nabla_b = [np.zeros_like(b) for b in self.biases]
        nabla_w = [np.zeros_like(w) for w in self.weights_transposed]
        mini_batch_cost = 0
        # Get X (num_features x num_examples)
        X = mini_batch[:, :-num_classes].T
        Y = mini_batch[:, -num_classes:]
        for i in range(mini_batch_size):
            # Make x column vector
            x = mini_batch[i, :-num_classes]
            y = mini_batch[i, -num_classes:]
            # Compute weight changes, bias changes, and loss for each training case
            delta_nabla_b, delta_nabla_w, training_example_cost = self.backpropagate(
                x, y)
            # Sum weight changes in each layer
            for r in range(self.num_layers):
                nabla_w[r] += delta_nabla_w[r]
                nabla_b[r] += delta_nabla_b[r]
            mini_batch_cost += training_example_cost
        # Update all weights and biases with the average gradient
        # self.weights_transposed = [w-(self.learning_rate/len(mini_batch))*nw
        #                 for w, nw in zip(self.weights_transposed, nabla_w)]
        # self.biases = [b-(self.learning_rate/len(mini_batch))*nb
        #                for b, nb in zip(self.biases, nabla_b)]
        for hl in range(self.num_layers):
            self.weights_transposed[hl] -= (self.learning_rate * nabla_w[hl]) / \
                mini_batch_size
            self.biases[hl] -= (self.learning_rate*nabla_b[hl])/mini_batch_size
        return mini_batch_cost/len(mini_batch)

    def backpropagate(self, x, y):
        """
        :type x: ndarray of shape num_features x ,
        :param x: training example

        :type y: ndarray of shape num_classes x ,
        :param y: target

        :returns: list of changes in weights for each layer, list of changes
        in biases for each layer, cost for training example
        """
        # Empty arrays to hold changes in each layer
        nabla_b = [np.zeros_like(b) for b in self.biases]
        nabla_w = [np.zeros_like(w) for w in self.weights_transposed]
        # Forward propagation
        output_layer = self.forward_propagation(x)
        loss_gradient = self.loss_type.gradient(y, output_layer)
        activation_gradient = self.activations[-1].gradient(self.zs[-1])
        delta = np.multiply(loss_gradient, activation_gradient)
        # Update last bias-layer
        nabla_b[-1] = delta
        # Update last weight-layer
        previous_a = self.activated_nodes[-2]
        nabla_w[-1] = np.outer(delta, previous_a)
        if isinstance(self.activations[-1], Softmax):
            # Get softmax jacobian of output values z (S by Z)
            J_softmax_by_output = self.layers[-1]['activation'].jacobian(
                output_values)
            # Add softmax-layer to jacobi-iteration (L by Z, S between)
            J_loss_by_layer = J_loss_by_layer @ J_softmax_by_output
        # Iterate backwards
        for l in range(2, self.num_layers+1):
            activation_gradient = self.activations[-l].gradient(self.zs[-l])
            # TODO: Check that this is ok
            layer_gradient = np.dot(self.weights_transposed[-l+1].T, delta)
            delta = np.multiply(layer_gradient, activation_gradient)
            nabla_b[-l] = delta
            previous_a = self.activated_nodes[-l-1]
            nabla_w[-l] = np.outer(delta, previous_a)
        prediction = self.activated_nodes[-1]
        cost = self.loss_type.apply_function(y, prediction)
        return nabla_b, nabla_w, cost

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
            z = np.dot(weights, activated_node)+bias
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

    def backpropagate_jacobi(self, x, y):
        # Empty arrays to hold changes in each layer
        nabla_b = [np.zeros_like(b) for b in self.biases]
        nabla_w = [np.zeros_like(w) for w in self.weights_transposed]
        # Forward propagation
        output_layer = self.forward_propagation(x)
        # Change in loss by change in output layer (L by Z, L by S for softmax)
        J_loss_by_layer = self.loss_type.gradient(y, output_layer)
        previous_a = self.activated_nodes[0]
        current_a = self.activated_nodes[1]
        J_layer_by_sum = np.multiply(np.identity(
            len(current_a)), self.activations[0].gradient(current_a))
        J_layer_by_weights = np.outer(previous_a, np.diag(J_layer_by_sum))
        nabla_w[0] = J_layer_by_weights.T

        output_errors = self.loss_type.apply_function(y, output_layer)
        return nabla_b, nabla_w, output_errors
