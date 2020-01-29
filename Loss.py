import numpy as np


class Loss:
    def __init__(self):
        pass

    def apply_function(self):
        raise NotImplementedError

    def gradient(self, values):
        raise NotImplementedError


class CrossEntropy:
    def __init__(self, activation):
        self.activation = activation

    def apply_function(self, y, s):
        """Return the cross-entropy of vectors y (target) and s (prediction)

        :type y: ndarray
        :param y: one-hot vector encoding correct class

        :type s: ndarray
        :param s: softmax vector

        :returns: scalar cost
        """
        # Since y is one-hot encoded, we can omit multiplying with it and just use s-values where y=1
        return -np.log(s[np.where(y)])

    def gradient(y, s):
        """
        Return the gradient of cross-entropy of vectors y and s

        :type y: ndarray
        :param y: one-hot vector encoding correct class

        :type s: ndarray
        :param s: softmax vector

        :returns: ndarray of size len(s)
        """
        return -y / s
