import configparser
import numpy as np

from Preprocess import *
from Activation import *
from Loss import *
from Network import *

"""
The following imports are OK, and not anything else: numpy, matplotlib.pyplot, configparser, enum,
sys and softmax from scipy.special. Notice that tanh is available from numpy.
"""

# Preprocessing, and reading from file
Preprocess = Preprocess()
"""
Preprocess.get_config_parameters('config.ini')
# Data
train_path = Preprocess.train_path
dev_path = Preprocess.dev_path
x_train, y_train = Preprocess.read_dataset(train_path)
x_dev, y_dev = Preprocess.read_dataset(dev_path)
# Model
activation_names = ['relu']
loss_type_name = 'L2'

hidden_layers = [2, 1]
activations = [Preprocess.get_activation(name) for name in activation_names]
loss_type = Preprocess.get_loss_type(loss_type_name)
# Hyper
learning_rate = 0.01
no_epochs = 100
L2_regularization = 'heihei'
"""


def write_weights_to_file(neural_network, path='somefile.txt'):
    with open(path, 'w') as filehandle:
        for weights in neural_network.weights_transposed:
            filehandle.write('{}\n'.format(weights))


if __name__ == '__main__':
    Menu = {
        -1: 'Testspace',
        1: 'Minimalist example for regression',
        2: 'Create config',
        3: 'Preprocess',
        4: 'Minimalist example for classification',
        5: 'Arbitrary NN',
    }[5]

    if Menu == 'Testspace':
        print('Welcome to testspace')

        hidden_layers = [2, 1]
        activations = [Preprocess.get_activation(
            name) for name in activation_names]
        loss_type = Preprocess.get_loss_type(loss_type_name)
        # Hyper
        learning_rate = 0.01
        no_epochs = 100
        L2_regularization = 'heihei'

        # X from dataset has shape no_examples x no_features
        # Y from dataset has shape no_examples x 1
        X = np.array([[1, 1],
                      [1, 0],
                      [0, 1],
                      [0, 0]])
        Y = np.array([1, 1, 1, 0])
        # Make Y a column vector
        Y = Y[:, np.newaxis]
        training_data = np.hstack((X, Y))
        if loss_type_name == 'L2':
            num_output_nodes = 1
        elif loss_type_name == 'cross_entropy':
            num_output_nodes = max(y_train) + 1
            y_train = Preprocess.one_hot_encode(
                num_examples_train, num_output_nodes, y_train)
            y_dev = Preprocess.one_hot_encode(
                num_examples_dev, num_output_nodes, y_dev)

        print('-----------------------------')
        print(X.shape)
        # Define network
        network = Network(hidden_layers[0])
        for i in range(len(hidden_layers)-1):
            network.add_layer(hidden_layers[i+1], activations[i])
        network.compile(learning_rate, loss_type)
        network.train(training_data)

    elif Menu == 'Minimalist example for regression':
        print('___ Task 2.1')
        # Hyper
        learning_rate = 0.01
        no_epochs = 100

        # X from dataset has shape num_examples x num_features
        # Y from dataset has shape num_examples x 1
        X = np.array([[1, 1],
                      [1, 0],
                      [0, 1],
                      [0, 0]])
        Y = np.array([1, 1, 1, 0])
        # Make Y a column vector
        Y = Y[:, np.newaxis]
        training_data = np.hstack((X, Y))

        x_dev = np.array([[1, 1],
                          [1, 0],
                          [0, 1],
                          [0, 0]])

        y_dev = np.array([1, 1, 1, 0])
        # Define network
        # First layer should have size num_features
        network = Network(X.shape[1])
        # Add output layer
        network.add_layer(1, Preprocess.get_activation('relu'))
        network.compile(learning_rate, Preprocess.get_loss_type('L2'))

        # Train network
        training_cost = network.train(training_data)
        print('--- training cost development:', training_cost)
        loss = network.test(x_dev, y_dev)
        print('-- validation loss:', loss)

        # Dump weights (transposed) to file
        write_weights_to_file(network)

    elif Menu == 'Create config':
        print('Creating config...')
        config = configparser.ConfigParser()
        config['DATA'] = {'training': './DATA/train_small.csv',
                          'validation': './DATA/validate_small.csv'}
        config['MODEL'] = {'layers': ' 24, 12,6',
                           'activations': 'relu, relu, tanh',
                           'loss_type': 'cross_entropy'}
        config['HYPER'] = {'learning_rate': '5.E-2',
                           'no_epochs': '10',
                           'L2_regularization': '5.E-3'}
        with open('config.ini', 'w') as configfile:
            config.write(configfile)

    elif Menu == 'Preprocess':
        print('Reading config file...')
        Preprocess = Preprocess()
        Preprocess.get_config_parameters('config.ini')

        path = './DATA/train_small.csv'
        x_train, y_train = Preprocess.read_dataset(path)
        no_examples = len(y_train)
        no_classes = max(y_train) + 1
        one_hot = Preprocess.one_hot_encode(no_examples, no_classes, y_train)

    elif Menu == 'Minimalist example for classification':
        print('___ Task 2.2')
        # Hyper
        learning_rate = 0.01
        no_epochs = 100

        # X from dataset has shape num_examples x num_features
        # Y from dataset has shape num_examples x 1
        X = np.array([[1, 1],
                      [1, 0],
                      [0, 1],
                      [0, 0]])
        Y = np.array([1, 2, 2, 0])

        num_output_nodes = Y.max()+1
        Y = Preprocess.one_hot_encode(X.shape[0], num_output_nodes, Y)

        # Make Y a column vector
        #Y = Y[:, np.newaxis]
        training_data = np.hstack((X, Y))

        x_dev = np.array([[1, 1],
                          [1, 0],
                          [0, 1],
                          [0, 0]])

        y_dev = np.array([1, 1, 1, 0])
        # Define network
        # First layer should have size num_features
        network = Network(X.shape[1])
        # Add output layer
        network.add_layer(num_output_nodes,
                          Preprocess.get_activation('softmax'))
        network.compile(
            learning_rate, Preprocess.get_loss_type('cross_entropy'))

        # Train network
        training_cost = network.train(training_data, num_output_nodes)
        print('--- training cost development:', training_cost)
        loss = network.test(x_dev, y_dev)
        print('-- validation loss:', loss)

        # Dump weights (transposed) to file
        write_weights_to_file(network)

    elif Menu == 'Arbitrary NN':
        # Data
        # train_path
        # dev_path

        # Model
        layers = [2]
        activations = ['relu']
        loss_type = 'L2'

        # Hyper
        learning_rate = 8e-2
        no_epochs = 5000
        L2_regularization = None

        # X from dataset has shape num_examples x num_features
        # Y from dataset has shape num_examples x 1
        X = np.array([[1, 1],
                      [1, 0],
                      [0, 1],
                      [0, 0]])
        Y = np.array([1, 0, 0, 0])

        # Dev sets
        x_dev = np.array([[1, 1]])  # ,
        #   [1, 0],
        #   [0, 1],
        #   [0, 0]])
        y_dev = np.array([1])  # , 0, 0, 0])

        num_classes = {'L2': 1, 'cross_entropy': 1+Y.max()}[loss_type]
        output_activation = {'L2': Preprocess.get_activation('tanh'),
                             'cross_entropy': Preprocess.get_activation('softmax')}[loss_type]

        if num_classes > 1:
            # One hot encode Y
            Y = Preprocess.one_hot_encode(X.shape[0], num_classes, Y)
            y_dev = Preprocess.one_hot_encode(
                x_dev.shape[0], num_classes, y_dev)
        else:
            # Make Y a column vector
            Y = Y[:, np.newaxis]
            y_dev = y_dev[:, np.newaxis]

        # Combine X and Y
        training_data = np.hstack((X, Y))

        # Define network
        # First layer should have size num_features
        network = Network(X.shape[1])
        # Add hidden layers
        num_hidden_layers = len(layers)
        if not(num_hidden_layers == 1 and layers[0] == 0):
            print('adding layers')
            for i in range(num_hidden_layers):
                layer_size = layers[i]
                activation = Preprocess.get_activation(activations[i])
                network.add_layer(layer_size, activation)
        network.add_layer(num_classes, output_activation)
        # Compile network
        network.compile(learning_rate, Preprocess.get_loss_type(loss_type))

        # Train network
        # TODO: DONE Forward propagation with x.shape: num_nodes x ,
        # TODO: BP with the same
        print('training data:', training_data)
        training_cost = network.train(
            training_data, num_classes, epochs=no_epochs, mini_batch_size=4)
        print('--- training cost development:', training_cost)
        print(network.activated_nodes[-1])
        # loss = network.test(x_dev, y_dev)
        # print('-- validation loss:', loss)

        # Dump weights (transposed) to file
        # write_weights_to_file(network)
