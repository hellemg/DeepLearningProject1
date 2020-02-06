import numpy as np


class Loss:
    def __init__(self):
        pass

    def apply_function(self, y, a):
        raise NotImplementedError

    def gradient(self, y, a):
        raise NotImplementedError


class CrossEntropy:
    def __init__(self):
        pass

    def __str__(self):
        return 'Crossentropy'

    def apply_function(self, y, a):
        raise NotImplementedError
        # Since y is one-hot encoded, we can omit multiplying with it and just use s-values where y=1
        return -np.log(a[np.where(y)])

    def gradient(self, y, a):
        raise NotImplementedError
        return -y / a


class L2:
    def __init__(self):
        pass

    def __str__(self):
        return 'L2'

    def apply_function(self, y, a):
        """
        :type a: ndarray of shape num_x x output_size
        :type y: ndarray of shape num_x x output_size

        :param a: prediction
        :param y: target

        :returns: ndarray of shape num_ex x output_size (output_size = 1 for L2)
        """
        return 1/2*(a - y)**2

    def gradient(self, y, a):
        """
        :type a: ndarray of shape num_x x output_size
        :type y: ndarray of shape num_x x output_size

        :param a: prediction
        :param y: target

        :returns: ndarray of shape num_ex x output_size
        """
        return a - y
