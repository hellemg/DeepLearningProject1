import configparser
import csv
import numpy as np

from Activation import *
from Loss import *


class Preprocess:
    def __init__(self):
        self.config = configparser.ConfigParser()
        # Data
        self.train_path = None
        self.dev_path = None
        # Model
        self.layers = None
        self.activations = None
        self.loss_type = None
        # Hyper
        self.learning_rate = None
        self.no_epochs = None
        self.L2_regularization = None

    def get_config_parameters(self, path, debug=False):
        self.config.read(path)

        # Data
        self.train_path = self.config['DATA']['training']
        self.dev_path = self.config['DATA']['validation']

        # Model
        # Read config-value, split into list based on comma, and remove whitespace and cast to int
        layers = self.config['MODEL']['layers'].split(',')
        self.layers = [int(x.strip(' ')) for x in layers]
        # TODO: if layers[0] == 0, no hidden layers
        self.activations = self.config['MODEL']['activations']
        self.loss_type = self.config['MODEL']['loss_type']
        """ TODO:
        Note that in
        addition to the mentioned functions, you will also need to handle the softmax as part of your
        implementation to handle the classification-loss (see below).
        """

        # Hyper
        self.learning_rate = self.config['HYPER']['learning_rate']
        self.no_epochs = self.config['HYPER']['no_epochs']
        self.L2_regularization = self.config['HYPER']['L2_regularization']
        print('... parameters are set')
        if debug:
            print('Sections in config:')
            for section in self.config.sections():
                print('-'+section)
                for key in self.config[section]:
                    print('--'+key)
                    print('----'+self.config[section][key])

    def read_dataset(self, path):
        """
        input: string, path for dataset to read
        returns:
        numpy array of shape no_examples x no_features, examples from dataset
        numpy array of shape no_examples x 1, labels from dataset
        """
        x = [[]]
        y = []
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                x.append([int(float(x)) for x in row[:-1]])
                y.append(int(float(row[-1])))
        print('... dataset is loaded from'+path)
        return np.array(x[1:]), np.array(y)

    def one_hot_encode(self, no_examples, no_classes, labels):
        """
        input:
        no_examples: int, number of examples
        no_classes: int, number of classes
        labels: numpy array of shape no_examples with ints, labels to encode
        returns:
        numpy array of shape no_examples, no_classes, one hot encoded label for each examples
        """
        one_hot = np.zeros((no_examples, no_classes))
        # Uses the i'th entry in each array at the same time
        one_hot[np.arange(no_examples), labels] = 1
        print('... labels are onehot encoded')
        return one_hot

    def get_activation(self, name):
        """
        Return the correct class of activation function given by name

        :type name: string
        :param name: name of activation function

        :returns: subclass of Activation
        """
        return {'relu': ReLU(), 'linear': Linear(),
                       'tanh': TanH(), 'softmax': Softmax()}[name]
    
    def get_loss_type(self, name):
        """
        Return the correct class of loss function given by name

        :type name: string
        :param name: name of activation function

        :returns: subclass of Loss
        """
        return {'L2': L2(), 'cross_entropy': CrossEntropy()}[name]
