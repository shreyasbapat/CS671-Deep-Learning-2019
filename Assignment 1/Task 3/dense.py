import numpy as np
#from numba import jit

from layers import Layer
from utils import cross_entropy

class DenseNetwork:

    def __init__(self):

        self.layers = [] # list containg layer objects
        self.num_layers = 0 # number of layers in network


    def input(self, dim):
        """Input Layer

        Parameters
        ----------
        dim: tuple
            Dimension of images
        activation: str
            Type of activation

        """
        input_n = dim[1] * dim[0]
        self.input_layer = Layer.input(input_n)
        self.layers.append(self.input_layer)
        self.num_layers += 1


    def add(self, n, activation):

        layer = Layer.from_prev_layer(n, self.layers[-1], activation)

        self.layers.append(layer)
        self.num_layers += 1


    def func_derivative(self, a, activation):
        if activation == 'sigmoid':
            sig_value = 1. / (1. + np.exp(-a))
            return sig_value * (1 - sig_value)
        elif activation == 'relu':
            if a >= 0:
                 return 1
            else:
                 return 0
        elif activation == 'one':
            return 0
        else:
            raise NotImplementedError("Not implemented as of now!")
       
       
    def derivative(self, vec, activation):
           derivative_vec = vec
           for i in range(np.size(vec)):
                  derivative_vec[i] = self.func_derivative(vec[i], activation)
           return derivative_vec

    def initialize(self):
        for i in range(1, self.num_layers):
            self.layers[i].init_layer()
       
    def forward_prop(self):
        for i in range(1, self.num_layers):
            self.layers[i].compile_layer(self.layers[i-1])

    #@jit(nopython=True, parallel=True)
    def back_prop(self, y, learning_rate):
        output_layer = self.layers[self.num_layers-1]
        output_dim = output_layer.n
        y_pred = output_layer.output
        #e_y = np.zeros((output_dim, 1)
        output_gradient = -1 * (y - y_pred)
        gradient = output_gradient # gradient for the last layer
        for i in range(1, self.num_layers):
                  
            # layer objects stored in layers of DenseNetwork
            prev_layer = self.layers[self.num_layers-i-1] # object of previous layer
            prev_values = prev_layer.output # output values in previous layer
            prev_input = prev_layer.values # input values in previous layer
            curr_layer = self.layers[self.num_layers-i] # current layer
            curr_weights = curr_layer.weights # current layer weights

            # gradients needed to update parameters
            weight_gradient = np.outer(gradient, prev_values)
            bias_gradient = gradient

            # calculation for calculating gradients for next iteration
            pre_activate_gradient = np.transpose(curr_weights).dot(gradient)
            derivative_vec = self.derivative(prev_input, prev_layer.activation)
            gradient = np.multiply(pre_activate_gradient, derivative_vec)

            # update the weights and bias in current layer
            curr_layer.update_params(weight_gradient, bias_gradient, learning_rate)


    def fit(self, x_train, y_train, max_epochs, learning_rate=0.002):
        self.initialize() # initialize all weights and biases of all layers
        for i in range(max_epochs):
            index = 0
            loss = 0
            # stochastic gradient descent with batch size = 1
            for x in x_train:
                self.layers[0].values = x
                self.layers[0].output = x
                y = y_train[index]
                self.forward_prop()
                self.back_prop(y, learning_rate)
                y_pred = self.layers[self.num_layers-1].output
                loss += cross_entropy(y, y_pred)
                index += 1

            print("iter no. : %d, loss func: %f" % (i, loss))


    def predict(self, x_test):
        y_pred = []
        for x in x_test:
            self.layers[0].values = x
            self.layers[0].output = x
            self.forward_prop()
            y_predict = np.argmax(self.layers[-1].output)
            y_pred.append(y_predict)
            
        n = len(y_pred)
        y_pred = np.array(y_pred)
        zer = np.zeros((n, 10))
        zer[np.arange(n), y_pred] = 1
        return zer