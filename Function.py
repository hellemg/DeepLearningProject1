class Function:
    def __init__(self):
        pass

    def backpropagate(self, next_der):
        # backpropagate partial derivative from next layer 
        """
        :type next_der: ndarray of shape num_features x num_examples
        """
        raise NotImplementedError

    def forward(self, x):
        # Forward propagate through layer
        raise NotImplementedError

    def apply_function(self):
        """
        :type z: ndarray of shape num_nodes x num_ex

        :returns: ndarray of shape num_nodes x num_ex
        """
        raise NotImplementedError

    def gradient(self, values):
        """
        :type z: ndarray of shape num_nodes x num_ex

        :returns: ndarray of shape num_nodes x num_ex
        """
        raise NotImplementedError
