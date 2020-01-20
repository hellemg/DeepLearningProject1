import configparser
import numpy as np
import csv

"""
The following imports are OK, and not anything else: numpy, matplotlib.pyplot, configparser, enum,
sys and softmax from scipy.special. Notice that tanh is available from numpy.
"""

if __name__ == '__main__':
    Menu = {
        -1: 'Testspace',
        1: 'Run',
        2: 'Create config',
        3: 'Read config',
        4: 'Transform data'
    }[4]

    if Menu == 'Testspace':
        print('Welcome to testspace')
        config = configparser.ConfigParser()
        config.read('config.txt')
        config.sections()

    elif Menu == 'Diamond Board':
        print('Welcome to AI programming project 1 :)')
        print('Running program...')

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

    elif Menu == 'Read config':
        print('Reading config file...')
        config = configparser.ConfigParser()
        config.read('config.ini')
        print(config.sections())

        for section in config.sections():
            print('---'+section)
            for key in config[section]:
                print(key, config[section][key])

        # Data
        train_path = config['DATA']['training']
        dev_path = config['DATA']['validation']

        # Model
        # Read config-value, split into list based on comma, and remove whitespace and cast to int
        layers = config['MODEL']['layers'].split(',')
        layers = [int(x.strip(' ')) for x in layers]
        # TODO: if layers[0] == 0, no hidden layers
        activations = config['MODEL']['activations']
        loss_type = config['MODEL']['loss_type']
        """ TODO:
        Note that in
        addition to the mentioned functions, you will also need to handle the softmax as part of your
        implementation to handle the classification-loss (see below).
        """

        # HYPER
        learning_rate = config['HYPER']['learning_rate']
        no_epochs = config['HYPER']['no_epochs']
        L2_regularization = config['HYPER']['L2_regularization']

    elif Menu == 'Transform data':
        print('Transforming data...')
        path = './DATA/train_small.csv'

        def read_dataset(path):
            """
            input: string, path for dataset to read
            returns:
            numpy array of shape no_examples x no_features, examples from dataset
            numpy array of shape no_examples x 1, labels from dataset
            """
            x = []
            y = []
            with open(path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    x.append([int(float(x)) for x in row[:-1]])
                    y.append(int(float(row[-1])))
            return np.array(x), np.array(y)

        def one_hot_encode(no_examples, no_classes, labels):
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
            one_hot[np.arange(no_examples), y_train] = 1
            return one_hot

        x_train, y_train = read_dataset(path)
        no_examples = len(y_train)
        no_classes = max(y_train) + 1
        one_hot = one_hot_encode(no_examples, no_classes, y_train)
