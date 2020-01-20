import numpy as np

class Function:
    def __init__(self):
        self.name = 'hei'

    def do_the_math(sself, value):
        raise NotImplementedError

    def derivative(self, value):
        """
        Return derivate of the value
        """
        raise NotImplementedError

class ReLU(Function):
    def __init__(self):
        super().__init__()

    def derivative(self, value):
        if value <= 0:
            return 0
        else:
            return 1

    def do_the_math(self, value):
        return np.maximum(0, value)

