import numpy as np

class LossFunc(object):
    """A class representing a loss function to be used to evaluate a 
    neural network's output. Objects belonging to this class
    hierarchy provide the two methods `compute` and `derive` which
    compute, respectively, the function's value and its derivative for a
    given pair of input (Y_ground_truth, Y_predicted).
    
    """
    def compute(self, Y, Y_hat):
        pass

    def derive(self, Y, Y_hat):
        pass

class CrossEntropy(LossFunc):

    def compute(self, Y, Y_hat):
        # Y is a row vector with shape (1, num. of training examples).
        m = Y.shape[1]
        return (-1. / m) * np.sum(np.multiply(Y, np.log(Y_hat))
                                  + np.multiply(1 - Y, np.log(1 - Y_hat)))

    def derive(self, Y, Y_hat):
        return np.divide(1 - Y, 1 - Y_hat) - np.divide(Y, Y_hat)

class MeanSquaredError(LossFunc):

    def compute(self, Y, Y_hat):
        # Y is a row vector with shape (1, num. of training examples).
        m = Y.shape[1]
        return (1. / m) * np.sum(np.square(Y - Y_hat))

    def derive(self, Y, Y_hat):
        return -2 * (Y - Y_hat)