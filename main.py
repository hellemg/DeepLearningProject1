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

def write_weights_to_file(neural_network, path='somefile.txt'):
    with open(path, 'w') as filehandle:
        for weights in neural_network.weights_transposed:
            filehandle.write('{}\n'.format(weights))


if __name__ == '__main__':
    Menu = {
        -1: 'Testspace',
        2: 'Create config',
        3: 'Preprocess',
        5: 'Arbitrary NN',
    }[5]

    if Menu == 'Testspace':
        print('Welcome to testspace')

    elif Menu == 'Create config':
        print('Creating config...')
        config = configparser.ConfigParser()
        config['DATA'] = {'training': './DATA/train_small.csv',
                          'validation': './DATA/validate_small.csv'}
        config['MODEL'] = {'layers': ' 512, 512,512',
                           'activations': 'relu, relu, relu',
                           'loss_type': 'cross_entropy'}
        config['HYPER'] = {'learning_rate': '1e-2',
                           'no_epochs': '50',
                           'L2_regularization': '0'}
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

    elif Menu == 'Arbitrary NN':
        # Data
        # print(train_path)
        # print(dev_path)
        preprocess = Preprocess()
        preprocess.get_config_parameters('config.ini')
        # # Data
        train_path = preprocess.train_path
        dev_path = preprocess.dev_path

        # Model
        # layers = [2]
        # activations = ['relu']
        # loss_type = 'L2'
        layers  = preprocess.layers
        activations = preprocess.activations
        loss_type = preprocess.loss_type

        # Hyper
        # learning_rate = 1e-1
        # no_epochs = 10000
        # L2_regularization = 0
        learning_rate = preprocess.learning_rate
        no_epochs = preprocess.no_epochs
        L2_regularization = preprocess.L2_regularization

        # X from dataset has shape num_examples x num_features
        # Y from dataset has shape num_examples x 1
        # X = np.array([[1, 1],
        #               [1, 0],
        #               [0, 1],
        #               [0, 0]])
        # Y = np.array([0, 1, 1, 0])
        X, Y = preprocess.read_dataset(train_path)

        # Dev sets
        # x_dev = X.copy()
        # y_dev = Y.copy()

        x_dev, y_dev = preprocess.read_dataset(dev_path)

        num_classes = {'L2': 1, 'cross_entropy': 1+Y.max()}[loss_type]
        output_activation = {'L2': preprocess.get_activation('tanh'),
                             'cross_entropy': preprocess.get_activation('softmax')}[loss_type]

        if num_classes > 1:
            # One hot encode Y
            Y = preprocess.one_hot_encode(X.shape[0], num_classes, Y)
            y_dev = preprocess.one_hot_encode(
                x_dev.shape[0], num_classes, y_dev)
        else:
            # Make Y a column vector
            Y = Y[:, np.newaxis]
            y_dev = y_dev[:, np.newaxis]

        # Combine X and Y
        training_data = np.hstack((X, Y))
        dev_data = np.hstack((x_dev, y_dev))

        # Define network
        # First layer should have size num_features
        network = Network(X.shape[1])
        # Add hidden layers
        num_hidden_layers = len(layers)
        if not(num_hidden_layers == 1 and layers[0] == 0):
            print('adding layers')
            for i in range(num_hidden_layers):
                layer_size = layers[i]
                activation = preprocess.get_activation(activations[i])
                network.add_layer(layer_size, activation)
        network.add_layer(num_classes, output_activation)
        # Compile network
        network.compile(learning_rate, preprocess.get_loss_type(loss_type))

        print(learning_rate)
        print(type(learning_rate))

        # Train network
        print('training data:', training_data)
        training_cost = network.train(
            training_data, num_classes, epochs=no_epochs, mini_batch_size=64, lbda=L2_regularization)
        #print('--- training cost development:', training_cost)
        #network.print_layers()
        loss = network.test(dev_data, num_classes)
        print('-- validation loss:', loss)

        # Dump weights (transposed) to file
        # write_weights_to_file(network)
