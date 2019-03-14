import numpy as np

from utils import relu, sigmoid

class Layer:

    def __init__(self, n, prev_n=0, prev_val=0, isinput=False, val=0, activation):
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
            self._prev_val = prev_val
            self.weights = None
            self.values = None
            self.activation = activation
            self.activate(activation)

        else:

            self.isinput = isinput
            self.n = n
            self.values = None
            self.activation = activation
            self.activate(activation)

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
        l_val = prev_layer.values
        return cls(n=n, prev_n=l_n, prev_val=l_val, activation=activation)

    @classmethod
    def input(cls, n, val, activation='relu'):
        """Return input `Layer`

        Parameters
        ----------

        n: int
            Number of neurons in input layer

        val: ~numpy.array
            Values of neurons of this layer

        """
        return cls(n=n, isinput=True, val=val, activation=activation)


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

    def calc_values(self):
        """
        Function to calculate values of neurons

        Returns
        -------
        values: ~numpy.array
            Matrix of W*prev_values

        """

        return self.weights.dot(self._prev_val)

    def input_values(self, val):

        return val

    def activate(self, activation='relu'):
        if activation=='relu':
            self.values = relu(self.values)
        elif activation=='sigmoid':
            self.values = sigmoid(self.values)
        else:
            raise NotImplementedError("No other activation function is implemented as of now!")

    # TODO : Incorporate Activation in the API (By taking it in the class methods!)
    # TODO : Work on Activation policy
