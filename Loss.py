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
        pass

    def __str__(self):
        return 'Crossentropy'

    def apply_function(self, y, s):
        """
        Return the cross-entropy loss of vectors y (target) and s (prediction)

        :type y: ndarray of shape num_classe x num_examples
        :param y: one-hot vector encoding correct class

        :type s: ndarray of shape num_classe x num_examples
        :param s: softmax vector

        :returns: scalar - cost
        """
        # Since y is one-hot encoded, we can omit multiplying with it and just use s-values where y=1
        return np.sum(-np.log(s[np.where(y)]))/s.shape[1]

    def gradient(self, y, s):
        """
        Return the gradient of cross-entropy of vectors y (target) and s (prediction after softmax)

        :type y: ndarray of shape num_classes x num_examples
        :param y: one-hot vector encoding correct class

        :type s: ndarray of shape num_classes x num_examples
        :param s: softmax vector

        :returns: ndarray of shape num_classes x num_examples
        """
        return -y / s


class L2:
    def __init__(self):
        pass

    def __str__(self):
        return 'L2'

    def apply_function(self, y, z):
        """
        Return the L2 loss of vectors y (target) and z (prediction)

        :type y: ndarray
        :param y: vector with target-values

        :type z: ndarray
        :param z: vector with predicted values

        :returns: scalar cost
        """
        output_size = z.shape[1]
        return np.sum((z-y)**2)/output_size

    def gradient(self, y, z):
        """
        Return the gradient of L2 of vectors y (target) and z (prediction) divided by batch_size

        :type y: ndarray of shape num_output_nodes x 1
        :param y: vector with target-values

        :type z: ndarray of shape num_output_nodes x num_examples
        :param z: vector with predicted values

        :returns: ndarray of size num_output_nodes x num_examples
        """
        output_size = z.shape[1]
        return 2*(z - y)/output_size
