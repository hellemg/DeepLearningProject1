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

if __name__ == '__main__':
    Menu = {
        -1: 'Testspace',
        1: 'Simple nn',
        2: 'Create config',
        3: 'Preprocess',
        4: 'Simple classifier',
    }[-1]

    if Menu == 'Testspace':
        print('Welcome to testspace')
        Preprocess = Preprocess()
        Preprocess.get_config_parameters('config.ini')
        # Get activation and loss classes
        # Model
        layers = [2, 1]
        activation_names = ['relu']
        loss_type_name = 'L2'
        activations = [Preprocess.get_activation(name) for name in activation_names]
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
        network = Network(layers[0])
        for i in range(len(layers)-1):
            network.add_layer(layers[i+1], activations[i])
        network.compile(learning_rate, loss_type)
        network.train(training_data)

    elif Menu == 'Simple nn':
        # TODO: Bias. Shapes. Batches.
        # TODO: Figure out shape of training examples, use batches. Ensure jacobi-iteration has correct shapes
        # TODO: Fix softmax-addition
        # TODO: Add bias to the sums
        # TODO: Add regularization
        print('Simple NN...')
        Preprocess = Preprocess()
        Preprocess.get_config_parameters('config.ini')
        # Get activation and loss classes
        # Model
        layers = [2, 3, 1]
        activation = Preprocess.get_activation('relu')
        loss_type = Preprocess.get_loss_type('L2')
        # Hyper
        learning_rate = 0.01
        no_epochs = 100
        L2_regularization = 'heihei'

        """     
        model = Model(learning_rate, loss_type)
        model.add_layer(2, activation, input_dim=2)
        # Training examples, one per row
        """
        X = np.array([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]])
        Y = np.array([[0], [1], [1], [1]])
        X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
        """
        X = np.array([[1], [1]])
        Y = np.array([[0.5], [0.5]])
        # Add x0 = 1 for the biases
        #ones = np.ones((X.shape[0], 1))
        #X = np.concatenate((ones, X), axis=1)
        #model.forward_propagation(np.array([[1], [1]]))
        Training examples usually are given as no_examples x no_features,
        slides says the opposite. no_features, no_examples, batch_size - find out all shapes
        Get help with the shapes of the Jacobians, go through each and write down what is needed
        Bias: should be affected by weights, but the slides don't have incoming weights on them?
        Regularization: explain
        model.fit(X, Y)
         """

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

        activation = Preprocess.get_activation('tanh')
        print(activation)
        loss_type = Preprocess.get_loss_type('L2')
        print(loss_type)

        path = './DATA/train_small.csv'
        x_train, y_train = Preprocess.read_dataset(path)
        no_examples = len(y_train)
        no_classes = max(y_train) + 1
        one_hot = Preprocess.one_hot_encode(no_examples, no_classes, y_train)
