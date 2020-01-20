import configparser


class FileReader:
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
        if debug:
            print('Sections in config:')
            for section in self.config.sections():
                print('-'+section)
                for key in self.config[section]:
                    print('--'+key)
                    print('----'+self.config[section][key])

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

    def read_dataset(self, path):
        """
        input: string, path for dataset to read
        returns:
        numpy array of shape no_examples x no_features, examples from dataset
        numpy array of shape no_examples x 1, labels from dataset
        """
        print('loading dataset from '+path)
        x = []
        y = []
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                x.append([int(float(x)) for x in row[:-1]])
                y.append(int(float(row[-1])))
        print('...dataset is loaded')
        return np.array(x), np.array(y)
