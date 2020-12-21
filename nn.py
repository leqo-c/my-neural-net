import numpy as np

from activations import ReLU, Sigmoid
from loss_functions import CrossEntropy
from metrics import accuracy


class MyNN(object):

    def __init__(self, layers_sizes, hidden_activation=ReLU(),
                 output_activation=Sigmoid(), cost_function=CrossEntropy()):
        """This class defines a custom Neural Network, consisting of one
        or more hidden layers.

        Args:
            layers_sizes (list): The number of units in each layer. Both
                the input and output layers' sizes must be specified (as
                the first and last entries, respectively) in addition to
                the hidden layers'.
            hidden_activation (`activations.ActivationFunction`,
                optional): The activation function to be used in all the
                hidden layers. Defaults to ReLU().
            output_activation (`activations.ActivationFunction`,
                optional): The activation function to be used in the
                output layer. Defaults to Sigmoid().
            cost_function (`loss_functions.LossFunc`, optional): The loss
                function to be used to evaluate the model's performances.
                Defaults to CrossEntropy().
        """
        self.layers_sizes = layers_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.parameters = self.initialize_params(layers_sizes)
        self.cost_function = cost_function

    def initialize_params(self, layers_sizes):
        """Initializes the model's internal parameters (i.e. the weights
        and biases for each layer, except for the input layer).

        Args:
            layers_sizes (list): The number of units in each layer. Both
                the input and output layers' sizes must be specified (as
                the first and last entries, respectively) in addition to
                the hidden layers'.

        Returns:
            dict: A dictionary containing the initial values of each
                layer's parameters.
        """
        parameters = dict()
        for i in range(1, len(layers_sizes)):
            parameters['W' + str(i)] = np.random.randn(
                layers_sizes[i], layers_sizes[i - 1]) * 0.01
            parameters['b' + str(i)] = np.zeros((layers_sizes[i], 1))

        return parameters

    def forward_pass(self, X):
        """Perform a forward pass through the network and compute the
        output.

        Args:
            X (numpy-array): The network's input data.

        Returns:
            tuple: The computed output (i.e. the outcome of the output
                layer) and a `cache` dictionary containing useful data
                for the subsequent backward pass.
        """
        cache = dict()
        cache['A0'] = X
        for i in range(1, len(self.layers_sizes)):
            Z = np.dot(self.parameters['W' + str(i)],
                       cache['A' + str(i-1)]) + self.parameters['b' + str(i)]
            activation = self.hidden_activation if i != len(
                self.layers_sizes) - 1 else self.output_activation
            A = activation.compute(Z)
            cache['Z' + str(i)] = Z
            cache['A' + str(i)] = A

        Y_hat = A

        return Y_hat, cache

    def backward_pass(self, Y, Y_hat, cache):
        """Perform a backward pass through the network, computing the
        derivatives of the W and b parameters of each layer.

        Args:
            Y (numpy-array): The ground truth outputs.
            Y_hat (numpy-array): The outputs computed through the
                forward pass.
            cache (dict): A dictionary containing useful variables for
                computing derivatives.

        Returns:
            dict: A `grads` dictionary containing the derivatives of
                each layer's parameters.
        """
        grads = dict()
        dA_prev = self.cost_function.derive(Y, Y_hat)
        m = Y.shape[1]

        for l in reversed(range(1, len(self.layers_sizes))):
            activation = (self.output_activation
                          if l == len(self.layers_sizes) - 1
                          else self.hidden_activation)
            dZ = dA_prev * activation.derive(cache['Z' + str(l)])
            grads['dW' + str(l)] = (1. / m) * \
                np.dot(dZ, cache['A' + str(l-1)].T)
            grads['db' + str(l)] = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
            dA_prev = np.dot(self.parameters['W' + str(l)].T, dZ)

        return grads

    def update_params(self, grads, learning_rate):
        """Perform the actual update of each layer's parameters by using
        the derivatives stored in `grads`.

        Args:
            grads (dict): A dictionary containing the derivatives of
                each layer's parameters.
            learning_rate (float): The step size to be used during the
                updates.
        """
        for l in range(1, len(self.layers_sizes)):
            self.parameters['W' + str(l)] -= learning_rate * \
                grads['dW' + str(l)]
            self.parameters['b' + str(l)] -= learning_rate * \
                grads['db' + str(l)]

    def fit(self, X, Y, epochs=1000, learning_rate=0.001,
            metric_function=accuracy):
        """Start the training process on the given training set.

        Args:
            X (numpy-array): The input features.
            Y (numpy-array): The ground-truth labels.
            epochs (int, optional): The number of training iterations.
                Defaults to 1000.
            learning_rate (float or function, optional): This value
                represents the step size to be used during the updates.
                Defaults to 0.001. You can alternatively pass a function
                that computes the learning rate as a function of the
                epoch number.
            metric_function (function, optional): A function that takes
                (ground-truth labels, predictions) and returns a score.
                Defaults to accuracy.
        """
        for i in range(epochs):
            Y_hat, cache = self.forward_pass(X)

            if i % 100 == 0:
                cost = self.cost_function.compute(Y, Y_hat)
                print(f'Loss: {cost}; Score: {metric_function(Y, Y_hat)}')

            grads = self.backward_pass(Y, Y_hat, cache)

            epoch = i + 1

            if type(learning_rate) != float:
                lr = learning_rate(epoch)
            else:
                lr = learning_rate

            self.update_params(grads, lr)

    def predict(self, X):
        """Perform a prediction on a set of features.

        Args:
            X (numpy-array): The input features you want to apply the
                predictions to.

        Returns:
            numpy-array: The network's predictions for the given input.
        """
        predictions, _ = self.forward_pass(X)
        return predictions
