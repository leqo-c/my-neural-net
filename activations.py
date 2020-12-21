import numpy as np


class ActivationFunction(object):
    """A class representing an activation function to be used
    in a neural network's hidden layers. Objects belonging to this class
    hierarchy provide the two methods `compute` and `derive` which
    compute, respectively, the function's value and its derivative on a
    given input.
    
    """

    def compute(self, X):
        pass
    def derive(self, X):
        pass

class Sigmoid(ActivationFunction):

    def compute(self, X):
        return 1 / (1 + np.exp(-X))

    def derive(self, X):
        s = self.compute(X)
        return s * (1 - s)

class Tanh(ActivationFunction):

    def compute(self, X):
        return np.tanh(X)

    def derive(self, X):
        return 1 - np.tanh(X)**2

class ReLU(ActivationFunction):

    def compute(self, X):
        return X * (X > 0).astype(int)

    def derive(self, X):
        return (X >= 0).astype(int)

class Linear(ActivationFunction):

    def compute(self, X):
        return X
    
    def derive(self, X):
        return np.ones_like(X)