import numpy as np
from numba import jit

from utils import relu, sigmoid, softmax

class Layer:

    def __init__(self, n, prev_n=0, prev_out=0, isinput=False, activation='relu'):
        """
        Constructor

        Parameters
        ----------
        n: int
            Number of neurons in the layer
        prev_n: int
            Number of layers in pervious layer
        prev_val: ~numpy.array
            Values of neurons in previous layer
        isinput: bool, Optional
            True if layer is input layer
        """
        if not isinput:

            self.isinput = isinput
            self.n = n
            self._prev_n = prev_n
            self._prev_out = prev_out
            self.weights = None
            self.bias = None
            self.values = None
            self.output = None
            self.activation = activation

        else:

            self.isinput = isinput
            self.n = n
            self.values = None
            self.output = None

    @classmethod
    def from_prev_layer(cls, n, prev_layer, activation):
        """Return `Layer` from previous layer logistics

        Parameters
        ----------

        n: int
            Number of neurons in this layer

        prev_layer: ~layers.Layer
            Previous Layer

        """
        l_n = prev_layer.n
        l_out = prev_layer.output
        return cls(n=n, prev_n=l_n, prev_out=l_out, activation=activation)

    @classmethod
    def input(cls, n):
        """Return input `Layer`

        Parameters
        ----------

        n: int
            Number of neurons in input layer

        val: ~numpy.array
            Values of neurons of this layer

        """
        return cls(n=n, isinput=True)

    @jit(nopython=True, parallel=True)
    def init_weights(self):
        """
        Function to initialise weights in a FC Layer

        Returns
        -------
        weights: ~numpy.array
            Randomly generated weights! (sigma * np.random.randn(...) + mu)

        """

        weights = 3 * np.random.randn(self.n, self._prev_n) + 1

        return weights

    @jit(nopython=True, parallel=True)
    def init_biases(self):
        """
        Function to initialise weights in a FC Layer

        Returns
        -------
        biases: ~numpy.array
            Randomly generated weights! (sigma * np.random.randn(...) + mu)

        """

        biases = 2 * (np.random.rand(self.n) - 0.5)

        return biases

    @jit(nopython=True, parallel=True)
    def calc_values(self):
        """
        Function to calculate values of neurons

        Returns
        -------
        values: ~numpy.array
            Matrix of W*prev_values

        """

        return self.weights.dot(self._prev_out) + self.bias

    def input_values(self, val):

        return val

    def activate(self, activation='relu'):
        if activation=='relu':
            self.output = relu(self.values)
        elif activation=='sigmoid':
            self.output = sigmoid(self.values)
        elif activation=='softmax':
            self.output = softmax(self.values)
        else:
            raise NotImplementedError("No other activation function is implemented as of now!")

    @jit(nopython=True, parallel=True)
    def init_layer(self):
        self.init_weights()
        self.init_biases()

    @jit(nopython=True, parallel=True)
    def compile_layer(self):
        self.values = self.calc_values()
        self.activate(self.activation)

    @jit(nopython=True, parallel=True)
    def update_params(self, W_gradient, b_gradient, learning_rate):
        self.weights = self.weights - learning_rate * W_gradient
        self.bias = self.bias - learning_rate * b_gradient
