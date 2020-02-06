import numpy as np


class Loss:
    def __init__(self):
        pass

    def apply_function(self, y, a):
        # Overwrites the comments in Function
        """
        :type a: ndarray of shape num_x x output_size
        :type y: ndarray of shape num_x x output_size

        :param a: prediction
        :param y: target

        :returns: ndarray of shape num_ex x output_size
        """
        raise NotImplementedError

    def gradient(self, y, a):
        # Overwrites the comments in Function
        """
        :type a: ndarray of shape num_x x output_size
        :type y: ndarray of shape num_x x output_size

        :param a: prediction
        :param y: target

        :returns: ndarray of shape num_ex x output_size
        """
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
        return 1/2*(y - a)**2

    def gradient(self, y, a):
        return y - a
