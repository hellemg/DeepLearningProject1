import configparser
import numpy as np
import matplotlib.pyplot as plt

from Preprocess import *
from Activation import *
from Loss import *
from Network import *
from Dense import *

np.random.seed(42)

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
        1: 'Create config',
        2: 'Arbitrary NN',
    }[2]

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
        with open('config_e5.ini', 'w') as configfile:
            config.write(configfile)

    elif Menu == 'Arbitrary NN':
        
        preprocess = Preprocess()
        preprocess.get_config_parameters('config.ini')
        
        # Data
        train_path = preprocess.train_path
        dev_path = preprocess.dev_path
        X, Y = preprocess.read_dataset(train_path)
        x_dev, y_dev = preprocess.read_dataset(dev_path)

        # Model
        layers  = preprocess.layers
        activations = preprocess.activations
        loss_type = preprocess.loss_type

        # Hyper
        learning_rate = preprocess.learning_rate
        no_epochs = preprocess.no_epochs
        L2_regularization = preprocess.L2_regularization

        # X from dataset has shape num_examples x num_features
        # Y from dataset has shape num_examples x 1
        # X = np.array([[1, 1],
        #               [1, 0],
        #               [0, 1],
        #               [0, 0]])
        # Y = np.array([1, 0, 0, 0])

        # Dev sets
        # x_dev = X.copy()
        # y_dev = Y.copy()

        # Get parameters for output layer
        num_classes = preprocess.get_num_classes(loss_type, Y)
        output_activation = preprocess.get_output_actication(loss_type)

        # Normalize
        X = X/np.max(X)
        # TODO: ask if max x_dev or max X?
        x_dev = x_dev/np.max(X)

        # Preprocess labels and combine data and labels to one array
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
        network = Network()
        num_hidden_layers = len(layers)
        input_dims = X.shape[1]
        if not(num_hidden_layers == 1 and layers[0] == 0):
            for i in range(num_hidden_layers):
                layer_size = layers[i]
                # Add dense layer
                dense = Dense(input_dims, layer_size)
                network.add_layer(dense)
                # Add activation layer
                activation = preprocess.get_activation(activations[i])
                network.add_layer(activation)
                input_dims = layer_size
        # Add output dense layer
        dense = Dense(input_dims, num_classes)
        network.add_layer(dense)
        # Add output activation layer
        network.add_layer(output_activation)

        # Compile network
        network.compile(learning_rate, preprocess.get_loss_type(loss_type), L2_regularization)

        # Train network
        training_cost, dev_cost = network.train(training_data, dev_data, num_classes, epochs=no_epochs, mini_batch_size=64)

        # Plot 
        plt.plot(np.arange(1,no_epochs+1), training_cost, 'r', label='training cost')
        plt.plot(np.arange(1,no_epochs+1), dev_cost, 'b', label='validation cost')
        plt.legend(loc="upper right")
        plt.show()

        print(np.argmin(dev_cost))

        # Dump weights (transposed) to file
        # write_weights_to_file(network)
