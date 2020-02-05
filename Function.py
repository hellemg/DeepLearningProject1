class Function:
    def __init__(self):
        pass

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
