import configparser

if __name__ == '__main__':
    Menu = {
        -1: 'Testspace',
        1: 'Run',
        2: 'Create config',
        3: 'Read config'
    }[3]

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

        # HYPER
        learning_rate = config['HYPER']['learning_rate']
        no_epochs = config['HYPER']['no_epochs']
        L2_regularization = config['HYPER']['L2_regularization']
