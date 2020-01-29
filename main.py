import configparser
import numpy as np

from Preprocess import *
from Function import *
from Model import *
from Loss import *

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
    }[1]

    if Menu == 'Testspace':
        print('Welcome to testspace')

    elif Menu == 'Simple nn':
        # TODO: Bias. Shapes. Batches.
        # TODO: Figure out shape of training examples, use batches. Ensure jacobi-iteration has correct shapes
        # TODO: Add bias to the sums
        # TODO: Add regularization
        print('Simple NN...')
        activation = ReLU()
        loss = L2()
        model = Model(learning_rate=0.05, loss_type=loss)
        model.add_layer(1, activation, input_dim=2)
        # Training examples, one per row
        X = np.array([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]])
        Y = [[0], [1], [1], [1]]
        # Add x0 = 1 for the biases
        #ones = np.ones((X.shape[0], 1))
        #X = np.concatenate((ones, X), axis=1)
        #model.forward_propagation(np.array([[1], [1]]))
        model.train(np.array([[1], [1]]), np.array([[1]]))
        
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
