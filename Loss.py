import numpy as np


class Loss:
    def __init__(self):
        pass

    def apply_function(self):
        raise NotImplementedError

    def gradient(self, values):
        raise NotImplementedError


class CrossEntropy:
    def __init__(self):
        print('... using crossentropy as loss function')

    def apply_function(self, y, s):
        """
        Return the cross-entropy loss of vectors y (target) and s (prediction)

        :type y: ndarray
        :param y: one-hot vector encoding correct class

        :type s: ndarray
        :param s: softmax vector

        :returns: scalar cost
        """
        # Since y is one-hot encoded, we can omit multiplying with it and just use s-values where y=1
        return -np.log(s[np.where(y)])

    def gradient(self, y, s):
        """
        Return the gradient of cross-entropy of vectors y (target) and s (prediction after softmax)

        :type y: ndarray
        :param y: one-hot vector encoding correct class

        :type s: ndarray
        :param s: softmax vector

        :returns: ndarray of size len(s)
        """
        return -y / s


class L2:
    def __init__(self):
        print('... using L2 as loss function')

    def apply_function(self, y, z):
        """
        Return the L2 loss of vectors y (target) and z (prediction)

        :type y: ndarray
        :param y: vector with target-values

        :type z: ndarray
        :param z: vector with predicted values

        :returns: scalar cost
        """
        output_size = len(z)
        return np.sum((y-z)**2)/output_size

    def gradient(self, y, z):
        """
        Return the gradient of L2 of vectors y (target) and z (prediction)

        :type y: ndarray of shape num_output_nodes x 1
        :param y: vector with target-values

        :type z: ndarray of shape num_output_nodes x num_examples
        :param z: vector with predicted values

        :returns: ndarray of size num_output_nodes x num_examples
        """
        output_size = len(z)
        return 2*(y - z)/output_size
