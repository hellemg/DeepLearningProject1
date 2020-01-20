import numpy as np

class Model:
    def __init__(self):
        self.layers = []
        #self.activations = activations
        self.loss_type = None
        self.weights = None
        self.architecture = None

    def train(self, data, labels, epochs):
        raise NotImplementedError

    def add_layer(self, no_nodes, activation, input_dim):
        weights = self.get_weights(no_nodes, input_dim)
        self.layers.append([weights, np.zeros((no_nodes, 1)), activation, input_dim])

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

    def forward_propagation(self, x_train):
        print('... forward propagation')
        prev_x = x_train
        for i, layer in enumerate(self.layers):
            print(layer)
            W_T = layer[0]
            x = layer[1]
            activation = layer[2]
            z = np.matmul(W_T, prev_x)
            self.layers[i][1] = activation.do_the_math(z)
            print(layer)
            prev_x = x


    """ 
    After setting up file-reader and parser for the config-file, the first task is to build a simple neural
    network as in Figure 1. In the figure there are 5 inputs, but this obviously depends on the inputdata. As the the model
    has no hidden layers, the output based on an input x using weights w is simply
    φ(w>x + b), with some nonlinearity φ. For now, use the ReLU, φ(z) = max (0, z).
    Optimize the weights to minimize the L2-loss. Here and for all the other learning tasks, start from
    a random initialization after first setting the seed by using np.random.seed(42
    """